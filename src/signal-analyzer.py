import pandas as pd
import sys
from os import makedirs
import matplotlib.pyplot as plt
import seaborn as sns

from src.km_clustering import cluster_analyzer
from src.ml_utils import elbow_cluster_number
from src.ms_clustering import meanshift_cluster_analyzer

""" This script  reads the data in CSV format 
    and estimates number of clusters using elbow method, and 
    performs MeanShift and K-Means clustering.
    The cluster labels then added to the dataset and saved as CSV file,
    figures are saved in the SVG format
"""

def read_input():
    data_name = ""
    if len(sys.argv) > 1 :
        file_name = sys.argv[1]       
    else:
        file_name = ""
    if len(file_name) < 2:
        file_name = input(' input file name: ')
    try:
        file_path = './data/' + file_name
        input_df = pd.read_csv(file_path)
        idx = file_name.find(".")
        if idx > 0 :
            data_name = file_name[:idx]
        else:
            data_name = file_name
        return input_df, data_name
    except FileNotFoundError :
        return [], data_name

fit_data, dataset_name = read_input()


# number of data points
sample_size = len(fit_data)
if sample_size > 0:
    output_dir = f'./results/{dataset_name}'
    makedirs(output_dir, exist_ok=True)

    """ clearing and preparing the data """
    if 'time' not in fit_data.columns:
        #index will be used as timestamp
        fit_data['time'] = fit_data.index
    #checking for default individual variables names
    valid_names = ['amp', 'fit_amp', 'amplitude', 't_rise', 't_dec', 't_decay']
    fit_vars = [col for col in fit_data.columns if col in valid_names]

    if len(fit_vars) == 0:
        #user input is needed
        user_names = input(f"pick the variables to fit from: \n {', '.join(fit_data.columns)} \n").split(',')
        fit_vars = [name.strip() for name in user_names]

    fit_data.dropna()
    drop_cols = [col for col in fit_data.columns if col not in fit_vars and col !='time' and col !='fit_label']
    fit_data.drop(columns=drop_cols, inplace=True)

    """ estimating number of clusters """
    opt_cluster_nums = elbow_cluster_number(fit_data, ind_vars=fit_vars, max_num=9, make_plots=True, dataset_name=dataset_name, save_plots=True)
    print(f'optimal cluster numbers by WCSS, C-H and D-B scores: {opt_cluster_nums} \n')

    """ MeanShift clustering 
        procedure will run first with default parameters and will repeat with user-defined
        parameters until user accepts the results
    """
    user_accept_fit = 'no'
    user_bandwidth = 0.2
    user_cluster_fraction = 0.96
    while 'y' not in user_accept_fit.strip().lower()[0:2]:
        ms_model, ms_moderated_labels, ms_cluster_num, ms_fig = meanshift_cluster_analyzer(
            fit_data, bandwidth_q=user_bandwidth, cuttof_fraction=user_cluster_fraction, ind_vars=fit_vars, time_var='time', plot_vars=[[1,2],[1,3]], dataset_name=dataset_name)
        print(f'cluster number by MeanShift method at bandwidth_q of {user_bandwidth} and main cluster fraction of  {user_cluster_fraction*100}% : {ms_cluster_num} ')
        user_accept_fit = input('accept fit (y/n)?')
        if len(user_accept_fit) == 0:
            user_accept_fit = 'yes'
        else:
            if 'n' in user_accept_fit.strip().lower()[0:1]:
                user_bandwidth = float(input('enter bandwidth_quantile : '))
                user_cluster_fraction = 0.01*float(input('enter minimal fraction of main clusters (%) : '))
            else:
                user_accept_fit = 'yes'
    fit_data['ms_label'] = ms_moderated_labels
    """ K-Means clustering """
    user_cluster_num = int(input('enter number of clusters for K-means procedure :'))
    km_model, km_fig = cluster_analyzer(
        fit_data, n_clusters=user_cluster_num, ind_vars=fit_vars, time_var='time', plot_vars=[[1,2],[1,3]], dataset_name=dataset_name)
    fit_data['Km_label'] = km_model.labels_

    # plotting time courses
    # user input is required for variable name, if none nothing will be plottet
    tmc_fig = None
    tms_var = input('enter the variable name for time course plot:\n')
    if tms_var in fit_vars:
        tmc_fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(15, 5))
        sns.set_context("paper")
        # sns.scatterplot(data=fit_data[fit_data.fit_label == 1], x='amp', y='t_decay', 
        #                     hue='Km_label', palette="deep",ax=axs[0,0])
        sns.scatterplot(data=fit_data, x='time', y=tms_var, 
                            hue='Km_label', palette="deep",ax=axs[0,0])
        axs[0,0].set_title('K-Means clusters')
        sns.scatterplot(data=fit_data, x='time', y=tms_var, 
                            hue='ms_label', palette="deep",ax=axs[0,1])
        axs[0,1].set_title('MeanShift clusters')
        tmc_fig.suptitle(f'{dataset_name} time course of cluster labels', fontsize=12)  
        plt.show()

    """ saving the results and figures 
        note! : files will be written over 
    """
    user_resp = input('save results (y/n)? ')
    save_res = (len(user_resp) > 0 and 'n' not in user_resp.strip().lower()[0:1]) or len(user_resp) == 0
    path_name = output_dir + f'/{dataset_name}_'
    fit_data.to_csv(path_name + 'res.csv', mode='w+')
    km_fig.savefig(path_name + '_Km_plt.svg', format='svg')
    ms_fig.savefig(path_name + '_ms_plt.svg', format='svg')
    if tmc_fig != None:
        tmc_fig.savefig(path_name + '_tmc_plt.svg', format='svg')
    
else:
    print ("file not found")

