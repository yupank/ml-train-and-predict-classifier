
import numpy as np
from numpy import linalg as la
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from src.ml_utils import make_mock_clusters, elbow_cluster_number
from src.km_clustering import cluster_analyzer
from src.opt_clustering import optics_cluster_analyser


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

""" testing randomly seeded signal clusters """
n_mock_clusters = 4

def test_random_clusters(elbow_max=9, cluster_plot=True, elbow_plot=True):
    """ this utility makes random n_mock_clusters and runs analyzer and checking functions """
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

    if elbow_max > 1:
        opt_nums = elbow_cluster_number(mock_sig_data,ind_vars=['amplitude','t_rise','t_decay'],max_num=9, make_plots=elbow_plot)
        print(f'optimal cluster numbers by WCSS, C-H and D-B scores: {opt_nums} ')
    if cluster_plot:
        test_clusters = [2,3,4,6]
        for clust_idx in test_clusters:
            res_model, fig = cluster_analyzer(mock_sig_data,ind_vars=['amplitude','t_rise','t_decay'],
                                            n_clusters=clust_idx, plot_vars=[['amplitude','t_rise'],['amplitude','t_decay']], dataset_name='Mock signals 1', n_mock=n_mock_clusters)
            accuracy= cluster_checker(res_model, mock_y,seed_centers)
            print(f'accuracy: {accuracy}')
            mock_sig_data['predicted_label']=res_model.labels_
            fig.savefig(f'./results/mock_signal_clustering_{n_mock_clusters}_{clust_idx}.svg',format='svg')

# test_random_clusters(cluster_plot=True, elbow_plot=True)

