import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.ml_utils import make_moon_clusters
from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from time import time


"""" prototyping of more advanced classificator based on small
        two-hidden layer neural network (implemented "manualy"  vs PyTorch-based)"""

""" creating non-linearly separable mock dataset """
# easy task - two 'moons'
# data_size = 2000
# mock_X, mock_y = make_moons(n_samples=data_size, noise=0.12, random_state=42)

""" helper function to visualize loss and accuracy during training """
def show_training_tracks(loss_track, accur_track, ds_name='moons', net_name='clusterer', trial_prefix='1'):
    fig, axs = plt.subplots(1,2, figsize = (10,4))
    axs[0].plot(range(len(loss_track)), loss_track)
    axs[0].set_title('loss over training cycle')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[1].plot(range(len(accur_track)), accur_track, 'r--')
    axs[1].set_title('accuracy over each 10 epochs')
    axs[1].set_xlabel('epoch X 10')
    fig.suptitle(f'training_cycle', fontsize=12)
    fig.savefig(f'./results/{ds_name}_{net_name}_{trial_prefix}.svg',format='svg')
    plt.show()

""" helper function to evaluate accuracy for outputs calculated as continouse values 
    Args: predicted_vals: numpy.array or torch.Tensor, true_y: array-like
    Returns: classification/labeling accuracy as  SkiLearn metrics
    prints classification report if corresponding flag is True
    plots predicted labels and errors if corresponding fla is True
"""
def report_label_accuracy(true_y, predicted_vals, print_report=False, plot_labels=False, X_data=None):
    pred_y = [1 if pred > 0.5 else 0 for pred in predicted_vals]
    acc_score = accuracy_score(true_y, pred_y)
    if print_report:
        print(classification_report(true_y, pred_y))
    # just for illustration purpose - to show students how accuracy looks like
    if plot_labels :
        fig_1, ax1 = plt.subplots(1,1)
        test_res_all = pd.DataFrame(X_data, columns=['X1','X2'])
        test_res_all['y_true'] = true_y
        test_res_all['label'] = pred_y
        test_res_neg = test_res_all[test_res_all['y_true'] != test_res_all['label']]
        sns.set_context("paper")
        sns.scatterplot(data=test_res_all, x='X1', y='X2', hue='label', palette="deep", ax=ax1)
        sns.scatterplot(data=test_res_neg, x='X1', y='X2', marker='x', s=55, c=test_res_neg['label'], cmap='Spectral', ax=ax1)
        fig_1.suptitle(f'clustering errors', fontsize=12)
        fig_1.savefig(f'./results/moon_&_blobs_test&errs_2.svg',format='svg')
        plt.show()
    return acc_score


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
                accuracy_track.append(report_label_accuracy(y, predictions))
            if i % 500 == 0:
                print(f'epoch_{i} loss:{loss_track[-1]} accuracy: {accuracy_track[-1]}')
            # back propagation
            self.find_gradient(X, y)
            self.update_step()
        return loss_track, accuracy_track
    

""" manually-implemented network """
def two_layer_clustering(data_size=2000, epochs_num=4000):
    # more difficult task - moons and blobs
    mock_X, mock_y = make_moon_clusters(int(data_size/2), data_size)
    # visual check
    fig_1, ax1 = plt.subplots(1,1)
    sns.set_context("paper")
    sns.scatterplot(data=mock_X, x=mock_X[:,0], y=mock_X[:,1], hue=mock_y, palette="deep", ax=ax1)
    fig_1.suptitle(f'mock data clusters', fontsize=12)
    fig_1.savefig(f'./results/moon_&_blobs_seeded_clusters.svg',format='svg')
    plt.show()

    # dividing data into trainig and test sets
    X_train, X_test, y_train, y_test = train_test_split(mock_X, mock_y, test_size=0.3, random_state=42)
    # reshaping lables into a column vector
    y_test = y_test.reshape(y_test.shape[0],1)
    y_train = y_train.reshape(y_train.shape[0],1)
    start_tm = time()
    end_tm = start_tm
    model = TwoLayerClassifier(h_layer_1_nodes=8, h_layer_2_nodes=4)
    loss_track, accur_track = model.train(X_train, y_train, lr=0.01, epochs=epochs_num)
    end_tm = time()
    show_training_tracks(loss_track, accur_track, net_name='2layer_man', trial_prefix='2')

    # testing the accuracy
    predictions = model.forward_pass(X_test)
    acc_score = report_label_accuracy(y_test, predictions, print_report=True, plot_labels=True, X_data=X_test)
    print(f'accuracy: {acc_score}')
    print(f'execution time: {end_tm-start_tm}')

# two_layer_clustering(data_size=4000, epochs_num=3000)

""" now the PyTorch-based model """
from torch import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchmetrics import Accuracy


class SeqNet(nn.Module):
    def __init__(self, inp_features=2, hl_1_nd=5, hl_2_nd=3):
        super().__init__()
        self.snn = nn.Sequential(
            nn.Linear(inp_features, hl_1_nd),
            nn.Linear(hl_1_nd, hl_2_nd),
            nn.Linear(hl_2_nd,1),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )
        
    def forward(self,x):
        out = self.snn(x)
        return out.squeeze(1)
        # return out


def nn_clusterer_training(data_X, data_y, n_epochs=200, bs=500, learn_rate = 0.001):
    train_dst = TensorDataset(torch.from_numpy(data_X).float(), torch.from_numpy(data_y).float())
    train_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=bs, shuffle=True)
    nnet = SeqNet()
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(nnet.parameters(), lr=learn_rate, momentum=0.95)
    loss_track = []
    accuracy_track = []
    # by some reason, metric.Accuracy does not work properly 
    # metric = Accuracy(task='multiclass', num_classes=2)
    
    # training cycle
    for ep in range(n_epochs):
        run_loss = 0.0
        run_accur = 0.0
        for count, (X, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = nnet(X)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            if ep%10 == 0:
                run_accur += report_label_accuracy(labels, output)
        # calculating and tracking accuracy over the epoch
        if ep%10 == 0:
            accuracy_track.append(run_accur/(count+1))
        # tracking the loss
        loss_track.append(run_loss/(count+1))

        if ep%50 == 0 :
            print(f'{ep} loss: {loss_track[-1]} accuracy: {accuracy_track[-1]}')
            # print(f'{ep} loss: {loss_track[-1]}')
    return nnet, loss_track, accuracy_track

""" training """
data_size = 4000
batch_size = 1000
test_size = 1000

moon_X, moon_y = make_moon_clusters(int(data_size/2), data_size)
start_tm = time()
model, loss_track, accur_track = nn_clusterer_training(moon_X, moon_y, n_epochs=300, bs=500)
end_tm = time()
show_training_tracks(loss_track, accur_track, net_name='Seq_ptr', trial_prefix='3')
print(f'execution time: {end_tm-start_tm}')


""" testing """

moon_X, moon_y = make_moon_clusters(int(test_size/2), test_size)
test_dst = TensorDataset(torch.from_numpy(moon_X).float(), torch.from_numpy(moon_y).int())
test_loader = torch.utils.data.DataLoader(dataset=test_dst, batch_size=len(test_dst), shuffle=True)
test_data = iter(test_loader)
test_X, test_y = next(test_data)
predictions = model(test_X)
# pred_y = [1 if pred > 0.5 else 0 for pred in predictions]
acc_score = report_label_accuracy(test_y, predictions, print_report=True, plot_labels=True, X_data=test_X)
print(f'accuracy: {acc_score}')