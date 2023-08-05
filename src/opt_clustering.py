import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import OPTICS, cluster_optics_dbscan
from ml_utils import make_mock_clusters

def optics_cluster_analyser(data, ind_vars=None, plot_vars=None, time_var=None, dbs_eps=0.9 ):
    """ the function performs unsupervised clasterisation of signals using OPTICS and DBSCAN methods
        Arguments: 
            data - dataFrame containing (as columns): timestamps (optionally) and parameters of signals as independent variables,
            no prior scalining is necessary
            optional args:   
            ind_vars - an array-like contining indeces or names of columns with independent variables, default - any column except one named 'time'
            plot_vars 2D-array withcolumn names/indices of [x,y] variables for plotting
            if None, consecuitive independent variables will be paired as x1,y1, x2,y2,...
            time_var - name of column containing timestamp, used for plotting the time course, if None or 'index' the row index will be used
        Returns: the fitted model containing the cluster labels (OPTICS) and reachibility and cluster scatterplots
    """
    selected_vars = [False for idx in range(data.columns.size)]
    if ind_vars == None:
        selected_vars = [col for col in data.columns if col != time_var]
    else:
        selected_vars = [ (idx in ind_vars) or (data.columns[idx] in ind_vars) for idx in range(data.columns.size)  ]
    X = data.loc[:,selected_vars]
    model = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05)
    model.fit(X)
    
    reachability = model.reachability_[model.ordering_]
    labels = model.labels_[model.ordering_]
    # print(model.cluster_hierarchy_)

    space = np.arange(len(X))
    labels_10 = cluster_optics_dbscan(
        reachability=model.reachability_, core_distances=model.core_distances_,
        ordering=model.ordering_, eps=dbs_eps)
    labels_20 = cluster_optics_dbscan(
        reachability=model.reachability_, core_distances=model.core_distances_,
        ordering=model.ordering_, eps=2*dbs_eps)
    # cluster numbers (there is always lable of -1 for noisy points):
    n_clusters = []
    n_clusters.append(len(np.unique(model.labels_))-1)
    n_clusters.append(len(np.unique(labels_10))-1)
    n_clusters.append(len(np.unique(labels_20))-1)
    print(f'n_clusters: {n_clusters}')

    fig = plt.figure(figsize=(10,6))
    grid = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(grid[0, :])
    ax2 = plt.subplot(grid[1,0])
    ax3 = plt.subplot(grid[1,1])
    ax4 = plt.subplot(grid[1,2])
    
    # reachability plot
    colors = ["g.", "m.", "b.", "y.", "c."]
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
    ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
    ax1.set_ylabel("Reachability (epsilon distance)")
    ax1.set_title("Reachability Plot")
    
    #OPTICS
    for klass, color in zip(range(0, 5), colors):
        Xk = X[model.labels_ == klass]
        ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], color, alpha=0.3)
        # print(Xk.head())
    # "noisy" points
    Xn = X[model.labels_ == -1]
    # ax2.plot(X[model.labels_ == -1, 0], X[model.labels_ == -1, 1], "k+", alpha=0.1)
    ax2.plot(Xn.iloc[:,0], Xn.iloc[:,1], "k+", alpha=0.1)
    ax2.set_title("Automatic Clustering\nOPTICS")

    # DBSCAN at 1xdbs_eps
    colors = ["g.", "r.", "b.", "c."]
    for klass, color in zip(range(0, 4), colors):
        Xk = X[labels_10 == klass]
        ax3.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], color, alpha=0.3)
    # "noisy" points
    Xn = X[model.labels_ == -1]
    ax3.plot(Xn.iloc[:,0], Xn.iloc[:,1], "k+", alpha=0.1)
    ax3.set_title(f"Clustering at {dbs_eps} epsilon cut\nDBSCAN")

    # DBSCAN at 1.5xdbs_eps.
    colors = ["g.", "m.", "y.", "c."]
    for klass, color in zip(range(0, 4), colors):
        Xk = X[labels_20 == klass]
        ax4.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], color, alpha=0.3)
    Xn = X[model.labels_ == -1]
    ax4.plot(Xn.iloc[:,0], Xn.iloc[:,1], "k+", alpha=0.1)
    ax4.set_title(f"Clustering at {2*dbs_eps} epsilon cut\nDBSCAN")

    plt.tight_layout()
    plt.show()
    return model, fig

n_points_per_cluster = 250
np.random.seed(0)
C1 = [-5, -2] + 0.8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + 0.1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + 0.3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
mock_X = np.vstack((C1, C2, C3, C4, C5, C6))
mock_data = pd.DataFrame(mock_X,columns=['amplitude','t_decay'])
optics_cluster_analyser(mock_data,dbs_eps=0.8)

mock_X,mock_y, seed_centers = make_mock_clusters(4)
mock_data = pd.DataFrame(mock_X,columns=['amplitude','t_rise','t_decay'])


mock_data.drop(columns='t_rise',inplace=True)
optics_cluster_analyser(mock_data,dbs_eps=0.8)

""" IMPORTANT !!! 
    testing OPTCIS methods with mock data at various epsilon values 
    showed much worse performance in detecting clusters than K-means;
    basically, optics worked well only when mock data formed clusters cleary visible by eye-balling (i.e.[C1-C6])
    but failed to detect clusters generated by make_mock_clusters function (more realistic), 
    which were confidently detected by k-mean clustering
"""
# mock_data['label']=mock_y