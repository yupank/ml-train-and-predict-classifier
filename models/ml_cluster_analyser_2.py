
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


from src.ml_utils import make_mock_clusters, elbow_cluster_number, cluster_checker
from src.km_clustering import cluster_analyzer



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

# more difficult clustering task - two moon-like clusters and two blobs

def test_moon_clusters(blob_size=600, moon_size=800, blob_ns=0.16, moon_ns=0.12, cluster_plot=True, elbow_plot=True):
    moon_X, moon_y = datasets.make_moons(n_samples=moon_size, shuffle=False, noise=moon_ns, random_state=42)
    blob_X, blob_y, = datasets.make_blobs(n_samples=[blob_size,blob_size],
                                            centers=[[-0.5,-0.5],[1.5,1]],cluster_std=blob_ns, 
                                            n_features=2,random_state=42)
    seed_centers = np.array([[0,1],[1,0.5],[-0.5,-0.5],[1.5,1]], dtype=float)
    blob_y = blob_y + 2
    mock_X = np.concatenate((moon_X, blob_X))
    mock_y = np.concatenate((moon_y, blob_y))
    mock_cols = ['X1','X2']
    mock_data = pd.DataFrame(mock_X, columns=mock_cols)
    mock_data['label'] = mock_y
    # visual check
    fig, ax1 = plt.subplots(1,1)
    sns.set_context("paper")
    sns.scatterplot(data=mock_X, x=mock_X[:,0], y=mock_X[:,1], hue=mock_y, palette="deep", ax=ax1)
    fig.suptitle(f'mock data clusters', fontsize=12)
    plt.show()
    if elbow_plot > 1:
        opt_nums = elbow_cluster_number(mock_data,ind_vars=mock_cols,max_num=9, make_plots=elbow_plot)
        print(f'optimal cluster numbers by WCSS, C-H and D-B scores: {opt_nums} ')
    if cluster_plot:
        test_clusters = [2,3,4,6]
    for clust_idx in test_clusters:
        res_model, fig = cluster_analyzer(mock_data,ind_vars=mock_cols, n_clusters=clust_idx, 
                                          plot_vars=[mock_cols], dataset_name='moons_&_blobs')
        mock_data['predicted_label']=res_model.labels_
        # accuracy = metrics.accuracy_score(mock_data['label'], mock_data['predicted_label'])
        accuracy= cluster_checker(res_model, mock_y,seed_centers)
        print(f'accuracy: {accuracy}')
        fig.savefig(f'./results/moon_&_blobs_pairs_into_{clust_idx}_clusters.svg',format='svg')
    
    """ NOTES: cluster analysis performs reasonably well when analyzing moon-like clusters,
                with overall accuracy about 0.8-0.85 and > 60% of 'moons' labled correctly """

# test_random_clusters(cluster_plot=True, elbow_plot=True)
test_moon_clusters(moon_ns=0.15)



