import pandas as pd
import sys
from os import makedirs

from src.km_clustering import cluster_analyzer
from src.ml_utils import elbow_cluster_number
from src.ms_clustering import meanshift_cluster_analyzer

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

""" clearing and preparing the data """
# number of data points
sample_size = len(fit_data)
if sample_size > 0:
    output_dir = f'./results/{dataset_name}'
    makedirs(output_dir, exist_ok=True)
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
    opt_cluster_nums = elbow_cluster_number(fit_data, ind_vars=fit_vars, max_num=9, make_plots=True, dataset_name=dataset_name, save_plots=True)
    print(f'optimal cluster numbers by WCSS, C-H and D-B scores: {opt_cluster_nums} ')
    user_bandwidth = 0.2
    ms_model, ms_moderated_labels, ms_cluster_num, ms_fig = meanshift_cluster_analyzer(
        fit_data, bandwidth_q=user_bandwidth , ind_vars=fit_vars, time_var='time', plot_vars=[[1,2],[1,3]], dataset_name=dataset_name)
    print(f'cluster number by MeanShift method at bandwidth quantile of {user_bandwidth} : {ms_cluster_num} ')
else:
    print ("file not found")

