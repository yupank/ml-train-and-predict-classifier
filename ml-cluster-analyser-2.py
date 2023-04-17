import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from math import ceil, floor, sqrt

def cluster_analyzer(data, n_clusters=2, ind_vars=None, plot_vars=None, time_var=None):
    """ the function performs unsupervised clasterisation of signals (optionally - time lapsed) using K-Means 
        Arguments: 
            data - dataFrame containing (as columns): timestamps (optionally) and parameters of signals as independent variables,
            no prior scalining is necessary
            n_clusters - expected number of clusters, can be estimated using ....
            optional args:   
            ind_vars - an array-like contining indeces or names of columns with independent variables, default - any column except one named 'time'
            x_vars and y_vars - arrays with aligned column names/indices of variables for plotting (i.e. y_vars[0] vs x_vars[0]),
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
    # checking index /column name to populate column names for plotting
    if plot_vars != None :
        # plot_name_x = [data.columns[idx] for idx in range(data.columns.size) if (idx in x_vars) or (data.columns[idx] in x_vars)]
        # plot_name_y = [data.columns[idx] for idx in range(data.columns.size) if (idx in y_vars) or (data.columns[idx] in y_vars)]
        plot_name_x = []
        plot_name_y = []
        for var in plot_vars:
            if var[0] in data.columns:
                plot_name_x.append(var[0])
            elif var[0] in range(data.columns.size):
                plot_name_x.append(data.columns[var[0]])
            if var[1] in data.columns:
                plot_name_y.append(var[1])
            elif var[1] in range(data.columns.size):
                plot_name_y.append(data.columns[var[1]])         
    else:
         plot_name_x = [data.columns[idx] for idx in range(data.columns.size) if idx % 2 == 0]
         plot_name_y = [data.columns[idx] for idx in range(data.columns.size) if idx % 2 == 1]
    
    if (len(plot_name_x)==0):
        plot_name_x = [data.columns[idx] for idx in range(data.columns.size) if idx % 2 == 0]
    if (len(plot_name_y)==0):
        plot_name_y = [data.columns[idx] for idx in range(data.columns.size) if idx % 2 == 1]

    #number of plots
    num_plots = max(len(plot_name_x), len(plot_name_y))
    # if x_ or y_ var list is shorter, previous value will be used
    if (len(plot_name_x) < num_plots ):
        plot_name_x.append(plot_name_x[-1])
    if (len(plot_name_y) < num_plots ):
        plot_name_y.append(plot_name_y[-1])
    print(plot_name_x)
    print(plot_name_y)
    # plotting
    plot_data = data.loc[:,selected_vars]
    plot_data['cluster_num'] = model.labels_
    plot_rows = floor(sqrt(num_plots))
    plot_cols = ceil(num_plots/plot_rows)
    print(num_plots, plot_rows, plot_cols)
    fig, axs = plt.subplots(plot_rows, plot_cols, squeeze=False, figsize=(4*plot_cols, 4*plot_rows))
    # sns.set_theme(style='darkgrid')
    sns.set_context("paper")
    for p_idx in range(num_plots):
        sns.scatterplot(data=plot_data, x=plot_name_x[p_idx], y=plot_name_y[p_idx], 
                        hue='cluster_num', palette="deep",ax=axs[p_idx//plot_cols, p_idx % plot_cols])
    plt.show()
    return model.labels_, fig


iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names']+['species'])
iris.columns = iris.columns.str.replace(' ',"_")
iris.species = iris.species.astype(np.int32)


iris['predicted_label']=cluster_analyzer(iris, n_clusters=3, ind_vars=[0,1,2,3],plot_vars=[[0,1],[2,3],[1,3]])[0]
print(iris.head())
print(iris.tail())
# cluster_analyzer(iris,ind_vars=['sepal_width_(cm)',  'petal_width_(cm)'])