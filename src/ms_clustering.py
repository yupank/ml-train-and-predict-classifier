import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import ceil, floor, sqrt

from sklearn.cluster import MeanShift, estimate_bandwidth
from ml_utils import make_mock_clusters, cluster_checker

def meanshift_cluster_analyzer(data,  ind_vars=None, bandwidth_q=0.15, plot_vars=None, time_var=None, dataset_name=''):
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
    fit_data = data.loc[:,selected_vars]

    bandwidth = estimate_bandwidth(fit_data,quantile=bandwidth_q)
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(fit_data)
    n_clusters = len(np.unique(model.labels_))

    #  plotting clusters
    # names of data columns actually used in clustering, needed to plot centers
    var_names = list(fit_data.columns.values)
    # checking index /column name to populate column names for plotting
    if plot_vars != None :
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
         plot_name_x = [fit_data.columns[idx] for idx in range(fit_data.columns.size) if idx % 2 == 0]
         plot_name_y = [fit_data.columns[idx] for idx in range(fit_data.columns.size) if idx % 2 == 1]
         
    if (len(plot_name_x)==0):
        plot_name_x = [fit_data.columns[idx] for idx in range(fit_data.columns.size) if idx % 2 == 0]
    if (len(plot_name_y)==0):
        plot_name_y = [fit_data.columns[idx] for idx in range(fit_data.columns.size) if idx % 2 == 1]
    #number of plots
    num_plots = max(len(plot_name_x), len(plot_name_y))
    # if x_ or y_ var list is shorter, previous value will be used
    if (len(plot_name_x) < num_plots ):
        plot_name_x.append(plot_name_x[-1])
    if (len(plot_name_y) < num_plots ):
        plot_name_y.append(plot_name_y[-1])

    # plotting
    plot_data = pd.DataFrame(data=data.loc[:,selected_vars])
    plot_data['cluster_num'] = model.labels_
    plot_rows = floor(sqrt(num_plots))
    plot_cols = ceil(num_plots/plot_rows) #extra column is needed for silhouette plot
    centers=model.cluster_centers_

    fig, axs = plt.subplots(plot_rows, plot_cols, squeeze=False, figsize=(4*plot_cols, 4*plot_rows))
    sns.set_context("paper")
    for p_idx in range(num_plots):
        row = p_idx//plot_cols
        col =p_idx % plot_cols
        sns.scatterplot(data=plot_data, x=plot_name_x[p_idx], y=plot_name_y[p_idx], 
                        hue='cluster_num', palette="deep",ax=axs[row, col])
        axs[row, col].scatter(centers[:, var_names.index(plot_name_x[p_idx])], 
                              centers[:, var_names.index(plot_name_y[p_idx])], c="r", s=25, marker='X' )
    fig.suptitle(f'{dataset_name} MeanShift fit for {n_clusters} clusters', fontsize=12)  
    plt.show()
    return model, n_clusters, fig
""" IMPORTANT 
    testing the MeanShift method with various mock data showed:
    1) accuracy of MS is lower than of K-Means (when called with right number of clusters);
    2) MS is very sensitive to "cluster_std", i.e. works well when clusters are "visually" separated;
    3) MS works better for lower number of clusters (e.g. in mock data)
    4) estimation of bandwidth is critical, estimation with higher quantile(>0.2) gives lower number of clusters than seeded,
        estimation with smaller quntiles ( < 0.1) give large number of "false" clusters;
        OPTIMAL quantile values lie within 0.1-0.2
"""

# testing with mock data

mock_X,mock_y, seed_centers = make_mock_clusters(3)
mock_sig_data = pd.DataFrame(mock_X,columns=['amplitude','t_rise','t_decay'])

# ms_model, n_clusters, fig = meanshift_cluster_analyzer(mock_sig_data, ind_vars=['amplitude','t_decay'],dataset_name='Mock_signals')
ms_model, n_clusters, fig = meanshift_cluster_analyzer(mock_sig_data, ind_vars=['amplitude','t_rise','t_decay'],plot_vars=[['amplitude','t_rise'],['amplitude','t_decay']],dataset_name='Mock_signals')
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
