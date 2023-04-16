import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_analyzer(data, n_clusters=2, ind_vars=None, x_vars=None,y_vars=None, time_var=None):
    """ the function performs unsupervised clasterisation of signals (optionally - time lapsed) using K-Means 
        Arguments: 
            data - dataFrame containing (as columns): timestamps (optionally) and parameters of signals as independent variables,
            no prior scalining is necessary
            n_clusters - expected number of clusters, can be estimated using ....
            optional args:   
            ind_vars - an array-like contining indeces or names of columns with independent variables, default - any column except one named 'time'
            x_vars and y_vars - arrays with aligned indeces/names of variables for plotting (i.e. y_vars[0] vs x_vars[0]),
            if None, conscuitive independent variables will be paired as x1,y1, x2,y2,...
            time_var - name of column containing timestamp, used for plotting the time course, if None or 'index' the row index will be used
        Returns: the dataframe contining the cluster lables (as int) aligned with the data index, cluster scatterplot and timecourse plot
    """
    selected_vars = [False for idx in range(data.columns.size)]
    if ind_vars == None:
        selected_vars = [col for col in data.columns if col != time_var]
    else:
        selected_vars = [ (idx in ind_vars) or (data.columns[idx] in ind_vars) for idx in range(data.columns.size)  ]
        # selected_vars = [var for var in ind_vars if var != time_var and (var in data.columns or (var >=0 and var < len(data.columns)))]

    X = data.loc[:,selected_vars] 
    print(X.head())
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=11, n_init='auto')
    model.fit(X)
    return model.labels_


iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names']+['species'])
iris.columns = iris.columns.str.replace(' ',"_")
iris.species = iris.species.astype(np.int32)


iris['predicted_label']=cluster_analyzer(iris, n_clusters=3, ind_vars=[0,1,2,3])
print(iris.head())
print(iris.tail())
# cluster_analyzer(iris,ind_vars=['sepal_width_(cm)',  'petal_width_(cm)'])