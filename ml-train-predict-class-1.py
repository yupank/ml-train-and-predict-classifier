import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.svm import SVC


def train_and_predict(train_input_features, train_outputs, prediction_features, method='log_reg'):
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
    if method == 'log_reg':
        # logistic regression model
        model = LogisticRegression(penalty='l2', solver='liblinear', C=10, random_state=0)
    elif method == 'des_tree':
        # decision tree
        print("decision tree")
        model = tree.DecisionTreeClassifier(criterion= 'entropy', random_state=0)
    elif method == 'svc' :
        # support vector machine
        model = SVC(kernel='linear', C=1.0, random_state=0)
    
    model.fit(train_input_features, train_outputs)
    prediction = model.predict(prediction_features)
    
    return prediction
def compare_models(X_train, X_test, y_train, y_test, dataset_name='iris'):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # now comparing different models
    y_pred = train_and_predict(X_train, y_train, X_test, method='log_reg')
    confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
    print("LogRes Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Test Confusion matrix :\n",confusion_matrix)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix).plot(ax=ax1)
    ax1.set_title('Logistic Regression')

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    y_pred = train_and_predict(X_train, y_train, X_test, method='des_tree')
    confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
    print("Tree Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Test Confusion matrix :\n",confusion_matrix)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix).plot(ax=ax2)
    ax2.set_title('Decision Tree Classifier')

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    y_pred = train_and_predict(X_train, y_train, X_test, method='svc')
    confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
    print("Tree Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Test Confusion matrix :\n",confusion_matrix)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix).plot(ax=ax3)
    ax3.set_title("Support Vector Machine")
    fig.suptitle(f'{dataset_name}  classification by different models - confusion matrices', fontsize=15)
    fig.savefig(f'./results/{dataset_name}_classifiers_1.svg',format='svg')
    plt.show()

# Iris classification
iris = datasets.load_iris()
X = iris.data
y = iris.target
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

compare_models(X_train, X_test, y_train, y_test)

# Digits classification
digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target
sc.fit(X)
X = sc.transform(X)
# Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False
# )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
compare_models(X_train, X_test, y_train, y_test, dataset_name='digits')

