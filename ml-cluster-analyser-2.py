import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import linalg as la
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from math import ceil, floor, sqrt

def make_mock_clusters(mock_clusters_num):
    # utility function generating mock signal data to test clustering and PCA:
    # normal distribution blobs (with overlap) and different sample size
    pop_size = 1000
    mock_centers = [[8+3*n,1.2+0.3*n*(n+1),8+n*n+3*n] for n in range(mock_clusters_num)]
    mock_stds = [1.3+0.3*n*n for n in range(mock_clusters_num)]
    mock_samples = [floor(pop_size*((1-n/mock_clusters_num))) for n in range(mock_clusters_num)]
    return datasets.make_blobs(n_samples=mock_samples, centers=mock_centers, cluster_std=mock_stds,n_features=3, random_state=42, return_centers=True)

def cluster_analyzer(data, n_clusters=2, ind_vars=None, plot_vars=None, time_var=None, dataset_name='', n_mock=None):
    """ the function performs unsupervised clasterisation of signals (optionally - time lapsed) using K-Means 
        Arguments: 
            data - dataFrame containing (as columns): timestamps (optionally) and parameters of signals as independent variables,
            no prior scalining is necessary
            n_clusters - expected number of clusters, can be estimated using ....
            optional args:   
            ind_vars - an array-like contining indeces or names of columns with independent variables, default - any column except one named 'time'
            plot_vars 2D-array withcolumn names/indices of [x,y] variables for plotting
            if None, conscuitive independent variables will be paired as x1,y1, x2,y2,...
            time_var - name of column containing timestamp, used for plotting the time course, if None or 'index' the row index will be used
        Returns: the fitted model containing the cluster labels, cluster scatterplot and timecourse plot
    """
    selected_vars = [False for idx in range(data.columns.size)]
    if ind_vars == None:
        selected_vars = [col for col in data.columns if col != time_var]
    else:
        selected_vars = [ (idx in ind_vars) or (data.columns[idx] in ind_vars) for idx in range(data.columns.size)  ]
    X = data.loc[:,selected_vars]

    model = KMeans(n_clusters=n_clusters, random_state=11, n_init='auto')
    model.fit(X)
    """ silhoutte analysis """
    silhouette_avg = metrics.silhouette_score(X, model.labels_)
    silhouette_values = metrics.silhouette_samples(X, model.labels_)
    cluster_silouhette_df = pd.DataFrame(columns=["score_value","y_value","cluster_num"])
    low_score_count = 0
    silplot_y_gap = 10
    silplot_y_up = 0
    silplot_y_labels = []
    for idx in range(n_clusters):
        # aggregating and sorting the scores for cluster = idx and preparing data for plot
        id_clust_silhouette_values = silhouette_values[model.labels_ == idx]
        id_clust_silhouette_values.sort()
        size = id_clust_silhouette_values.shape[0]
        silplot_y_labels.append({"x":max(-0.05,id_clust_silhouette_values.min()),
                                 "y":silplot_y_up+(idx+1.5)*silplot_y_gap,"text":str(idx)})
        id_clust_y_values = np.linspace(silplot_y_up+idx*silplot_y_gap,silplot_y_up+size,num=size)
        id_clust_labels = np.zeros(size,dtype=int) + idx
        silplot_y_up += size
        id_silh_df = pd.DataFrame({"score_value":id_clust_silhouette_values,"y_value":id_clust_y_values,"cluster_num":id_clust_labels})
        cluster_silouhette_df=pd.concat([cluster_silouhette_df,id_silh_df], ignore_index=True)
        max_score = id_clust_silhouette_values.max()
        print(f'cluster {idx} scores - mean: {id_clust_silhouette_values.mean()} max: {max_score}')
        if max_score < silhouette_avg:
            low_score_count += 1
    print(f'average silhouette score for {n_clusters} clusters: {silhouette_avg}, {low_score_count} lay fully below average ')

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
    # plotting
    plot_data = pd.DataFrame(data=data.loc[:,selected_vars])
    plot_data['cluster_num'] = model.labels_
    plot_rows = floor(sqrt(num_plots))
    plot_cols = ceil(num_plots/plot_rows)+1 #extra column is needed for silhouette plot
    centers=model.cluster_centers_
    # names of data columns actually used in clustering, needed to plot centers
    var_names = list(X.columns.values)
    fig, axs = plt.subplots(plot_rows, plot_cols, squeeze=False, figsize=(4*plot_cols, 4*plot_rows))
    # sns.set_theme(style='darkgrid')
    sns.set_context("paper")
    for p_idx in range(num_plots):
        row = p_idx//plot_cols
        col =p_idx % plot_cols
        sns.scatterplot(data=plot_data, x=plot_name_x[p_idx], y=plot_name_y[p_idx], 
                        hue='cluster_num', palette="deep",ax=axs[row, col])
        axs[row, col].scatter(centers[:, var_names.index(plot_name_x[p_idx])], 
                              centers[:, var_names.index(plot_name_y[p_idx])], c="r", s=25, marker='X' )
   #plotting the silhouettes 
    silh_ax = axs[0,plot_cols-1]
    silh_ax.set_xlim([-0.1,1])
    sns.lineplot(data=cluster_silouhette_df, x='score_value',y='y_value', 
                 hue='cluster_num', palette='deep', legend=False, ax=silh_ax)
    silh_ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    silh_ax.set_xlabel('The silhouette coefficient')
    silh_ax.set_yticks([])
    silh_ax.set_ylabel('Cluster label')
    #labeling the silhouettes
    for label in silplot_y_labels:
        silh_ax.text(label['x'],label['y'],label['text'])
    if n_mock != None:
        fig.suptitle(f'{dataset_name} K-means fit for {n_mock} mock_clusters', fontsize=12) # for mock testing
    else:
        fig.suptitle(f'{dataset_name} K-means fit for {n_clusters} clusters', fontsize=12)  # real data analysis
    plt.show()
    return model, fig


def cluster_checker(fit_model, cluster_labels, cluster_seeds=None):
    """ utility function to check the accuracy of cluster labeling 
        Arguments: 
            fitted model (KMeans), 
            cluster_labels - an array-like labels for the data used to fit the model
            cluster_seeds - array of parameters of claster centers used to generate data (e.g. used for making blobs),
            needed to align fit_model.lables_ with lables for seeded clusters, picking the nearest one, 
            if None, labels will be compared as they are
        Returns: accuracy calculated using sklearn.metrics or -1 if parameters are incorrect 
    """
    # align lables
    if cluster_seeds.shape[0] == fit_model.cluster_centers_.shape[0]:
        nearest_centers = [ np.argmin([ la.norm(np.subtract(seed_coord, fit_coord)) for fit_coord in fit_model.cluster_centers_]) 
                           for seed_coord in cluster_seeds ]
        aligned_seeds = np.choose(cluster_labels, nearest_centers).astype(np.int32)
        if len(cluster_labels) == len(aligned_seeds):
            return metrics.accuracy_score(aligned_seeds, fit_model.labels_)
        else:
            return -1
    else:
        return metrics.accuracy_score(cluster_labels, fit_model.labels_)

""" testing iris classification """

iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names']+['species'])
iris.columns = iris.columns.str.replace(' ',"_")
iris.species = iris.species.astype(np.int32)
# res_model, fig = cluster_analyzer(iris, n_clusters=3, ind_vars=[0,1,2,3],plot_vars=[[0,1],[2,3],[1,3]],dataset_name='Iris')
# iris['predicted_label']=res_model.labels_
# print(iris.head())
# fig.savefig(f'./results/iris_clustering_1.svg',format='svg')
n_mock_clusters = 4
""" testing randomly seeded signal clusters """
mock_X,mock_y, seed_centers = make_mock_clusters(n_mock_clusters)
mock_sig_data = pd.DataFrame(mock_X,columns=['amplitude','t_rise','t_decay'])
mock_sig_data['label']=mock_y
# visual check
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# sns.scatterplot(data=mock_sig_data, x='amplitude', y='t_rise', hue='label', palette="deep",ax=ax1)
# sns.scatterplot(data=mock_sig_data, x='amplitude', y='t_decay', hue='label', palette="deep",ax=ax2)
# fig.suptitle(f'mock signals seeded clusters', fontsize=12)
# plt.show()
# fig.savefig(f'./results/mock_signal_seeded_clusters.svg',format='svg')


test_clusters = [2,3,4,6]
for clust_idx in test_clusters:
    res_model, fig = cluster_analyzer(mock_sig_data,ind_vars=['amplitude','t_rise','t_decay'],
                                    n_clusters=clust_idx, plot_vars=[['amplitude','t_rise'],['amplitude','t_decay']], dataset_name='Mock signals 1', n_mock=n_mock_clusters)
    accuracy= cluster_checker(res_model, mock_y,seed_centers)
    print(f'accuracy: {accuracy}')
    mock_sig_data['predicted_label']=res_model.labels_
    fig.savefig(f'./results/mock_signal_clustering_{n_mock_clusters}_{clust_idx}.svg',format='svg')

# print(mock_sig_data.head(10))
# print(mock_sig_data.tail(10))