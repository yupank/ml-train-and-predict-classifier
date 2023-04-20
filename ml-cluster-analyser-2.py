import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import linalg as la
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from math import ceil, floor, sqrt

def cluster_analyzer(data, n_clusters=2, ind_vars=None, plot_vars=None, time_var=None, dataset_name=''):
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
    plot_data = data.loc[:,selected_vars]
    plot_data['cluster_num'] = model.labels_
    plot_rows = floor(sqrt(num_plots))
    plot_cols = ceil(num_plots/plot_rows)
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
    fig.suptitle(f'{dataset_name} clustering', fontsize=12)
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
    if cluster_seeds.shape == fit_model.cluster_centers_.shape :
        nearest_centers = [ np.argmin([ la.norm(np.subtract(seed_coord, fit_coord)) for fit_coord in fit_model.cluster_centers_]) 
                           for seed_coord in cluster_seeds ]
        aligned_seeds = np.choose(cluster_labels, nearest_centers).astype(np.int32)
        if len(cluster_labels) == len(aligned_seeds):
            return metrics.accuracy_score(aligned_seeds, fit_model.labels_)
        else:
            return -1
        # check_df = pd.DataFrame(fit_model.labels_, columns=['predicted_label'])
        # check_df['aligned_seed_label'] = aligned_seeds
        # print(check_df.head(20))

    else:
        return -1

""" testing iris classification """

iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names']+['species'])
iris.columns = iris.columns.str.replace(' ',"_")
iris.species = iris.species.astype(np.int32)
# res_model, fig = cluster_analyzer(iris, n_clusters=3, ind_vars=[0,1,2,3],plot_vars=[[0,1],[2,3],[1,3]],dataset_name='Iris')
# iris['predicted_label']=res_model.labels_
# print(iris.head())
# fig.savefig(f'./results/iris_clustering_1.svg',format='svg')

""" testing randomly seeded signal clusters """
# generating mock signal data to test clustering and PCA:
# normal distribution blobs (with overlap) and different sample size
mock_clusters_num = 4
pop_size = 1000
mock_centers = [[8+3*n,1.2+0.3*n*(n+1),8+n*n+3*n] for n in range(mock_clusters_num)]
mock_stds = [1+0.2*n*n for n in range(mock_clusters_num)]
mock_samples = [floor(pop_size*((1-n/mock_clusters_num))) for n in range(mock_clusters_num)]
mock_X,mock_y, seed_centers = datasets.make_blobs(n_samples=mock_samples, centers=mock_centers, cluster_std=mock_stds,n_features=3, random_state=42, return_centers=True)

mock_sig_data = pd.DataFrame(mock_X,columns=['amplitude','t_rise','t_decay'])
mock_sig_data['label']=mock_y
# visual check
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# sns.scatterplot(data=mock_sig_data, x='amplitude', y='t_rise', hue='label', palette="deep",ax=ax1)
# sns.scatterplot(data=mock_sig_data, x='amplitude', y='t_decay', hue='label', palette="deep",ax=ax2)
# fig.suptitle(f'mock signals seeded clusters', fontsize=12)
# plt.show()
# fig.savefig(f'./results/mock_signal_seeded_clusters.svg',format='svg')

res_model, fig = cluster_analyzer(mock_sig_data,ind_vars=['amplitude','t_rise','t_decay'],
                                  n_clusters=4, plot_vars=[['amplitude','t_rise'],['amplitude','t_decay']], dataset_name='Mock signals 1')
accuracy= cluster_checker(res_model, mock_y,seed_centers)
print(f'accuracy: {accuracy}')
mock_sig_data['predicted_label']=res_model.labels_
fig.savefig(f'./results/mock_signal_1_clustering_1.svg',format='svg')

# print(mock_sig_data.head(10))
# print(mock_sig_data.tail(10))