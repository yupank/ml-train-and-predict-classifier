import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from ml_utils import make_mock_clusters, cluster_checker

def meanshift_cluster_analyzer(data,  ind_vars=None, plot_vars=None, time_var=None, dataset_name='', n_mock=None):
    """ the function performs unsupervised clasterisation of signals using MeanShift method 
        Arguments: 
            data - dataFrame containing (as columns): timestamps (optionally) and parameters of signals as independent variables,
            no prior scalining is necessary
            n_clusters - expected number of clusters, can be estimated using ....
            optional args:   
            ind_vars - an array-like contining indeces or names of columns with independent variables, default - any column except one named 'time'
            plot_vars 2D-array withcolumn names/indices of [x,y] variables for plotting
            if None, consecuitive independent variables will be paired as x1,y1, x2,y2,...
            time_var - name of column containing timestamp, used for plotting the time course, if None or 'index' the row index will be used
        Returns: the fitted model containing the cluster labels, number of clusters and cluster scatterplot
    """
    selected_vars = [False for idx in range(data.columns.size)]
    if ind_vars == None:
        selected_vars = [col for col in data.columns if col != time_var]
    else:
        selected_vars = [ (idx in ind_vars) or (data.columns[idx] in ind_vars) for idx in range(data.columns.size)  ]
    X = data.loc[:,selected_vars]

    bandwidth = estimate_bandwidth(X,quantile=0.15)
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(X)
    n_clusters = len(np.unique(model.labels_))

    fig = plt.figure(figsize=(10,6))
    plt.clf()
    colors = ["g.", "m.", "b.", "y.", "c."]
    markers = ["x", "o", "^"]
    labels = model.labels_
    for k in range(n_clusters):
        members= labels == k
        cluster_center = model.cluster_centers_[k]
        plt.plot(X.iloc[members, 0], X.iloc[members,1], colors[k%5],markers[k%3])
        plt.plot(cluster_center[0], cluster_center[1], markers[k%3], markeredgecolor="k",markersize=14)
    
    plt.show()
    return model, n_clusters, fig

# testing with mock data
centers = [[1,1],[-1,-1], [1, -1]]
# mock_X, mock_y, seed_centers = make_blobs(n_samples=3000, centers=centers, cluster_std=0.8, return_centers=True)
# mock_sig_data = pd.DataFrame(mock_X,columns=['amplitude','t_decay'])
mock_X,mock_y, seed_centers = make_mock_clusters(3)
mock_sig_data = pd.DataFrame(mock_X,columns=['amplitude','t_rise','t_decay'])

ms_model, n_clusters, fig = meanshift_cluster_analyzer(mock_sig_data)
mock_sig_data['predicted_label']=ms_model.labels_
mock_sig_data['label']=mock_y

print(f'Estimated number of clusters: {n_clusters} ')
accuracy = cluster_checker(ms_model, mock_y,seed_centers)
print(f'accuracy: {accuracy}')

# visual check
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.scatterplot(data=mock_sig_data, x='amplitude', y='predicted_label', hue='label', palette="deep",ax=ax1)
sns.scatterplot(data=mock_sig_data, x='t_decay', y='predicted_label', hue='label', palette="deep",ax=ax2)
fig.suptitle(f'mock signals seeded clusters', fontsize=12)
plt.show()
