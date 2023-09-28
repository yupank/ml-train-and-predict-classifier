import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from time import time


"""" prototyping of more advanced classificator based on small
        two-hidden layer neural network (implemented "manualy"  vs PyTorch-based)"""

""" creating non-linearly separable mock dataset """
# easy task - two 'moons'
# data_size = 2000
# mock_X, mock_y = make_moons(n_samples=data_size, noise=0.12, random_state=42)

# more difficult task - moons and blobs
data_size = 1600
moon_X, moon_y = make_moons(n_samples=data_size, shuffle=True, noise=0.15, random_state=42)
blob_X, blob_y, = make_blobs(n_samples=[int(data_size/2),int(data_size/2)],
                                        centers=[[-0.7,-0.3],[1.7,0.8]],cluster_std=0.2, 
                                        n_features=2,random_state=42)
mock_X = np.concatenate((moon_X, blob_X))
mock_y = np.concatenate((moon_y, blob_y))

# visual check
fig, ax1 = plt.subplots(1,1)
sns.set_context("paper")
sns.scatterplot(data=mock_X, x=mock_X[:,0], y=mock_X[:,1], hue=mock_y, palette="deep", ax=ax1)
fig.suptitle(f'mock data  clusters', fontsize=12)
plt.show()

# dividing data into trainig and test sets
X_train, X_test, y_train, y_test = train_test_split(mock_X, mock_y, test_size=0.2, random_state=42)
# reshaping lables into a column vector
y_test = y_test.reshape(y_test.shape[0],1)
y_train = y_train.reshape(y_train.shape[0],1)

""" activation helper """
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

""" binary cross-entropy loss helper """
def bss_loss(Y, Y_target):
    bss = - np.mean( ( (Y * np.log(Y_target)) + ((1-Y)*np.log(1-Y_target)) ) )
    return bss

""" the two-hidden layer NN model class """
class TwoLayerClassifier:
    def __init__(self, inp_features=2, h_layer_1_nodes=5, h_layer_2_nodes=3, lr = 0.01):
        """ performs random initilization of weight and bias matrices"""
        self.output_nodes = 1
        #default learning rate
        self.lr = lr 
        # connecting input layer to 1st hidden layer
        self.wt_1 = np.random.rand(inp_features, h_layer_1_nodes)
        self.bs_1 = np.zeros((1, h_layer_1_nodes))
        # connecting  1st hidden layer to 2nd hidden layer
        self.wt_2 = np.random.rand(h_layer_1_nodes, h_layer_2_nodes)
        self.bs_2 = np.zeros((1, h_layer_2_nodes))
        # connecting 2nd hidden layer to output layer
        self.wt_3 = np.random.rand(h_layer_2_nodes, self.output_nodes)
        self.bs_3 = np.zeros((1, self.output_nodes))
        # self.gr_wt_1 = None
        # self.gr_wt_2 = None
        # self.gr_wt_3 = None
        # self.gr_bs_1 = None

    
    def forward_pass(self, X):
        """ calculates activation values for each layer and returns the predicted output"""
        # activation of 1st hidden layer
        z_1 = np.dot(X, self.wt_1) + self.bs_1
        self.A_1 = np.tanh(z_1)
        # activation of 2nd hidden layer
        z_2 = np.dot(self.A_1, self.wt_2) + self.bs_2
        self.A_2 = np.tanh(z_2)
        # output (prediction)
        z_3 = np.dot(self.A_2, self.wt_3) + self.bs_3
        self.A_3 = sigmoid(z_3)
        return self.A_3
    
    def find_gradient(self, X, Y):
        """ methods calculates adn stores gradients"""
        xs = X.shape[0]
        delta_3 = self.A_3 - Y
        self.gr_wt_3 = np.dot(self.A_2.T, delta_3)/xs
        self.gr_bs_3 = np.sum(delta_3, axis=0, keepdims=True)/xs

        delta_2 = np.dot(delta_3, self.wt_3.T) * (1-np.power(self.A_2,2))
        self.gr_wt_2 = np.dot(self.A_1.T, delta_2)/xs
        self.gr_bs_2 = np.sum(delta_2, axis=0, keepdims=True)/xs

        delta_1 = np.dot(delta_2, self.wt_2.T)*(1-np.power(self.A_1,2))
        self.gr_wt_1 = np.dot(X.T, delta_1)/xs
        self.gr_bs_1 = np.sum(delta_1, axis=0, keepdims=True)/xs
    
    def update_step(self):
        """ makes one step of weights and bias update """
        self.wt_1 -= self.lr * self.gr_wt_1
        self.bs_1 -= self.lr * self.gr_bs_1

        self.wt_2 -= self.lr * self.gr_wt_2
        self.bs_2 -= self.lr * self.gr_bs_2

        self.wt_3 -= self.lr * self.gr_wt_3
        self.bs_3 -= self.lr * self.gr_bs_3

    def train(self, X, y, lr=0.01, epochs=1000):
        """ performs training cycle and returns the report of loss and accuracy for each epoch """
        loss_track = []
        accuracy_track = []
        for i in range(epochs):
            # forward pass
            predictions = self.forward_pass(X)
            loss_track.append(bss_loss(y, predictions))
            if i % 10 == 0:
                y_pred = [1 if pred > 0.5 else 0 for pred in predictions]
                accuracy_track.append(accuracy_score(y, y_pred))
            if i % 500 == 0:
                print(f'epoch_{i} loss:{loss_track[-1]} accuracy: {accuracy_track[-1]}')
            # back propagation
            self.find_gradient(X, y)
            self.update_step()
        return loss_track, accuracy_track

start_tm = time()
end_tm = start_tm
model = TwoLayerClassifier(h_layer_1_nodes=8, h_layer_2_nodes=4)
# print(f'w2: {model.wt_2}')
# print(f'b2: {model.bs_2}')

loss_track, accur_track = model.train(X_train, y_train, lr=0.01, epochs=4000)
end_tm = time()
fig, axs = plt.subplots(1,2, figsize = (10,4))
axs[0].plot(range(len(loss_track)), loss_track)
axs[0].set_title('loss over training cycle')
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('loss')
axs[1].plot(range(len(accur_track)), accur_track, 'r--')
axs[1].set_title('accuracy over each 10 epochs')
axs[1].set_xlabel('epoch X 10')
plt.show()

# testing the accuracy


predictions = model.forward_pass(X_test)
y_pred = [1 if pred > 0.5 else 0 for pred in predictions]

print(classification_report(y_test, y_pred))
print(f'accuracy: {accuracy_score(y_test, y_pred)}')
print(f'execution time: {end_tm-start_tm}')