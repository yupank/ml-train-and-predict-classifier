import matplotlib.pyplot as plt
import numpy as nm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_and_predict(train_input_features, train_outputs, prediction_features):
    """
    :param train_input_features: (numpy.array) A two-dimensional NumPy array where each element
                        is an array containing features(e.g. sepal length, sepal width, petal length, and petal width)   
    :param train_outputs: (numpy.array) A one-dimensional NumPy array where each element
                        is a number representing the catergorie (e.g. species of iris) which is described in
                        the same row of train_input_features, e.g 0 - Iris setosa, 1 - Iris versicolor, 2 - Iris virginica.
    :param prediction_features: (numpy.array) A two-dimensional NumPy array where each element
                        is an array that contains features (e.g sepal length, sepal width, petal length, and petal width)
    :returns: (list) The function should return an iterable (like list or numpy.ndarray) of the predicted 
                        iris species, one for each item in prediction_features
    """   
    # l1 regularization gives better results
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=10, random_state=0)
    lr.fit(train_input_features, train_outputs)
    prediction = lr.predict(prediction_features)
    return prediction

iris = datasets.load_iris()
X = iris.data
y = iris.target
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25, random_state=0)

y_pred = train_and_predict(X_train, y_train, X_test)
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Test Confusion matrix :\n",confusion_matrix)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix).plot(ax=ax2)
plt.show()


