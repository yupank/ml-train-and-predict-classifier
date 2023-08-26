import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import ceil, floor, sqrt

from sklearn.cluster import MeanShift, estimate_bandwidth


def meanshift_cluster_analyzer(data,  ind_vars=None, bandwidth_q=0.15, cuttof_fraction=0.95, plot_vars=None, time_var=None, dataset_name=''):
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
        Returns: the fitted model containing the cluster labels, moderated labels and number of clusters, and cluster scatterplot.
        IMPORTANT !! the cluster labels will be moderated, as compared to the standard output of MeanShift().
        Analysis of real biological signals with MS typically gives few large clusters and several "outlier" clusters 
        containing 1-3 data points. As a remedy, cluster numbers will be moderated based on assumption 
        the that 'main' clusters should comprise significant fraction (by default 95%) of data samples, 
        the "outlier" clusters with just a few data points will be piled together in the last cluster
    """
    """ IMPORTANT ! 
    testing the MeanShift method with various mock data showed:
    1) accuracy of MS is lower than of K-Means (when called with right number of clusters);
    2) MS is very sensitive to "cluster_std", i.e. works well when clusters are "visually" separated;
    3) MS works better for lower number of clusters (e.g. in mock data)
    4) estimation of bandwidth is critical, estimation with higher quantile(>0.2) gives lower number of clusters than seeded,
        estimation with smaller quntiles ( < 0.1) give large number of "false" clusters;
        OPTIMAL quantile values lie within 0.1-0.2
    """
    selected_vars = [False for idx in range(data.columns.size)]
    if ind_vars == None:
        selected_vars = [col for col in data.columns if col != time_var]
    else:
        selected_vars = [ (idx in ind_vars) or (data.columns[idx] in ind_vars) for idx in range(data.columns.size)  ]
    fit_data = data.loc[:,selected_vars]
    data_size = len(fit_data)
    bandwidth = estimate_bandwidth(fit_data,quantile=bandwidth_q)
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=500)
    model.fit(fit_data)
    """ moderating the cluster labels  """
    # clust_labels_tot = np.unique(model.labels_)
    # n_clusters_tot = len(clust_labels_tot)
    clusters_df = pd.DataFrame(data=data.iloc[:,1])
    clusters_df['initial_label'] = model.labels_

    # grouping clusters and calulating relative frequencies
    cluster_sizes = clusters_df.groupby('initial_label', as_index=False).count()
    cluster_sizes.columns=['initial_label', 'count']
    cluster_sizes['count'] = cluster_sizes['count']/data_size

    # sorting and calculating cumulative score to find the cut-off cluster number 
    cluster_sizes.sort_values('count', inplace=True, ascending=False, ignore_index=True)
    cluster_sizes['cum_score'] = cluster_sizes['count'].cumsum()

    # number of clusters to be used for moderating lables
    n_clusters_mod = cluster_sizes[cluster_sizes.cum_score > cuttof_fraction]['initial_label'].idxmin()+1
    print(f'n_clusters = {n_clusters_mod}')
    clusters_df['moderated_label']=clusters_df['initial_label'].apply(
        lambda lab: lab if lab < n_clusters_mod else n_clusters_mod
    )
    # print(clusters_df.sample(25))

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
    plot_data['cluster_num'] = clusters_df.moderated_label

    plot_rows = floor(sqrt(num_plots))
    plot_cols = ceil(num_plots/plot_rows) #extra column is needed for silhouette plot
    centers=model.cluster_centers_[:n_clusters_mod]

    fig, axs = plt.subplots(plot_rows, plot_cols, squeeze=False, figsize=(4*plot_cols, 4*plot_rows))
    sns.set_context("paper")
    for p_idx in range(num_plots):
        row = p_idx//plot_cols
        col =p_idx % plot_cols
        sns.scatterplot(data=plot_data, x=plot_name_x[p_idx], y=plot_name_y[p_idx], 
                        hue='cluster_num', palette="deep",ax=axs[row, col])
        axs[row, col].scatter(centers[:, var_names.index(plot_name_x[p_idx])], 
                              centers[:, var_names.index(plot_name_y[p_idx])], c="r", s=25, marker='X' )
    fig.suptitle(f'{dataset_name} MeanShift fit for {n_clusters_mod} clusters', fontsize=12)  
    plt.show()
    return model, clusters_df.moderated_label, n_clusters_mod, fig




