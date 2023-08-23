import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from math import floor

def make_mock_clusters(mock_clusters_num):
    # utility function generating mock signal data to test clustering and PCA:
    # normal distribution blobs (with overlap) and different sample size
    pop_size = 1000
    mock_centers = [[8+3*n,1.2+0.3*n*(n+1),8+n*n+3*n] for n in range(mock_clusters_num)]
    mock_stds = [1.3+0.3*n*n for n in range(mock_clusters_num)]
    mock_samples = [floor(pop_size*((1-n/mock_clusters_num))) for n in range(mock_clusters_num)]
    return make_blobs(n_samples=mock_samples, centers=mock_centers, cluster_std=mock_stds,n_features=3, random_state=42, return_centers=True)


def cluster_checker(fit_model, cluster_labels, cluster_seeds=None):
    """ utility function to check the accuracy of cluster labeling 
        Arguments: 
            fitted model (KMeans or MeanShift), 
            cluster_labels - an array-like labels for the data used to fit the model
            cluster_seeds - array of parameters of claster centers used to generate data (e.g. used for making blobs),
            needed to align fit_model.lables_ with lables for seeded clusters, picking the nearest one, 
            if None, labels will be compared as they are
        Returns: accuracy calculated using sklearn.metrics or -1 if parameters are incorrect
    """
    # align lables
    # print(f'shapes 0: {cluster_seeds.shape[0]} - {fit_model.cluster_centers_.shape[0]}')
    # print(f'shapes 1: {cluster_seeds.shape[1]} - {fit_model.cluster_centers_.shape[1]}')
    if cluster_seeds.shape[1] == fit_model.cluster_centers_.shape[1]:
        nearest_centers = [ np.argmin([ la.norm(np.subtract(seed_coord, fit_coord)) for fit_coord in fit_model.cluster_centers_]) 
                           for seed_coord in cluster_seeds ]
        aligned_seeds = np.choose(cluster_labels, nearest_centers).astype(np.int32)
        if len(cluster_labels) == len(aligned_seeds):
            return metrics.accuracy_score(aligned_seeds, fit_model.labels_)
        else:
            return -1
    else:
        return metrics.accuracy_score(cluster_labels, fit_model.labels_)

# looking for optimal cluster number with elbow method
def find_elbow(values, x_vals= None, gradient=-1):
    """ utility function to find the elbow in the array of values
        Args:
            values - an iterable of floats/integers,
            x_vals (optional) - array of x-values, where elbow position is to be found,
            if None, x_vals will be calculated as number of clusters, staritng from 2
            gradient (optional) - a sign of supposed gradient of values against x_vals,
            default is descending (-1)
        Returns: an elbow position, 
            determined as x_vals at index where decrement/increment in values starts to decline
            or None if values is empty
    """
    if len(values) > 0:
        if x_vals != None:
            x_num = x_vals
        else:
            x_num = range(2,len(values)+2)
        opt_idx = 1
        delta_new = (values[1]-values[0])*gradient
        delta_prev = 0
        while opt_idx < len(values)-1 and delta_new >= delta_prev:
            delta_prev = delta_new
            opt_idx += 1
            delta_new = (values[opt_idx] - values[opt_idx-1])*gradient
        opt_num = x_num[opt_idx]
        return opt_num
    else:
        return None
    

def elbow_cluster_number(data, ind_vars=None, time_var=None,max_num=5, make_plots=False, dataset_name=None, save_plots=False):
    """ Utility function to evaluate optimal number of clusters by elbow method,
        using three criteria of clustering performance (WCSS, C-H index and D-B index) 
    Args:   
        data - dataFrame containing (as columns): timestamps (optionally) and parameters of signals as independent variables,
        optional args:   
            ind_vars - an array-like contining indeces or names of columns with independent variables, default - any column except one named 'time'
            time_var - name of column containing timestamp, this column cannot be used as independent variable
        ind_vars=None, plot_vars=None, time_var=None,
        max_num - maximal number of clusters to check, within range(2,max_num), default is 5

    Returns: y-data for optimal number of clusters
    Plots: the elbow plot of error vs cluster number
    """
    # preparing the data for K-means fit
    selected_vars = [False for idx in range(data.columns.size)]
    if ind_vars == None:
        selected_vars = [col for col in data.columns if col != time_var]
    else:
        selected_vars = [ (idx in ind_vars) or (data.columns[idx] in ind_vars) for idx in range(data.columns.size)  ]
    X = data.loc[:,selected_vars]


    cluster_num = range(2,max_num+1)
    # conventional "within "within-cluster sum-of-squares" criterion
    inertia = []
    # Calinski-Harabasz Index
    ch_index = []
    # Davies-Bouldin Index
    db_index = []
    for n in cluster_num :
        # model = KMeans(featuresCol='standardized',k=n).fit(data_scale_output)
        model = KMeans(n_clusters=n, random_state=11, n_init='auto')
        model.fit(X)
        inertia.append(model.inertia_)
        ch_index.append(metrics.calinski_harabasz_score(X,model.labels_))
        db_index.append(metrics.davies_bouldin_score(X,model.labels_))
        # errors.append(model.summary.trainingCost)    
    wcss_num = find_elbow(inertia,x_vals=cluster_num)
    ch_num = find_elbow(ch_index,x_vals=cluster_num, gradient=1)
    db_num = find_elbow(db_index,x_vals=cluster_num)
    if make_plots :
        fig = plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.plot(cluster_num, inertia)
        plt.title('Inertia score')
        plt.subplot(1,3,2)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('C-H index')
        plt.plot(cluster_num, ch_index)
        plt.title('Calinski-Harabasz score')
        plt.subplot(1,3,3)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('D-B index')
        plt.plot(cluster_num, db_index)
        plt.title('Davies-Bouldin score')
        if dataset_name != None:
            fig.suptitle(f'Clustering Scores for {dataset_name}',fontsize=12)
        plt.show()
        if save_plots:
            fig.savefig(f'./results/{dataset_name}/{dataset_name}_DBCH_scores.svg',format='svg')
    return wcss_num, ch_num, db_num