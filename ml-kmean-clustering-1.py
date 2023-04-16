import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

""" this script illusrates the basic pricinples of usage of 
K-Means method of unsupervised learning (clustering) """

# Iris clustering
iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names']+['species'])
iris.columns = iris.columns.str.replace(' ',"_")
iris.species = iris.species.astype(np.int32)

X = iris.iloc[:,:4]
print(X.head())
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
y = iris.species

# K-Means Cluster
model = KMeans(n_clusters=3, random_state=11, n_init='auto')
model.fit(X)
# print(model.labels_)
# allinging model output labels with labels index from dataset: 0 and 1 need swapping, 2 is fine :
iris['pred_species'] = np.choose(model.labels_, [1,0,2]).astype(np.int32)
print(iris.head())
print(iris.tail())
print("Accuracy :", metrics.accuracy_score(iris.species, iris.pred_species))
print("Classification report :\n", metrics.classification_report(iris.species, iris.pred_species))

# Set the size of the plot
plt.figure(figsize=(10,7))
# Create a colormap for red, green and blue
cmap = ListedColormap(['r', 'g', 'b'])
# Plot Sepal
plt.subplot(2, 2, 1)
plt.scatter(iris['sepal_length_(cm)'], iris['sepal_width_(cm)'], c=cmap(iris.species), marker='o', s=50)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Sepal (Actual)')
plt.subplot(2, 2, 2)
plt.scatter(iris['sepal_length_(cm)'], iris['sepal_width_(cm)'], c=cmap(iris.pred_species), marker='o', s=50)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Sepal (Predicted)')
plt.subplot(2, 2, 3)
plt.scatter(iris['petal_length_(cm)'], iris['petal_width_(cm)'], c=cmap(iris.species),marker='o', s=50)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width(cm)')
plt.title('Petal (Actual)')
plt.subplot(2, 2, 4)
plt.scatter(iris['petal_length_(cm)'], iris['petal_width_(cm)'], c=cmap(iris.pred_species),marker='o', s=50)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

plt.title('Petal (Predicted)')
plt.tight_layout()
plt.show()

