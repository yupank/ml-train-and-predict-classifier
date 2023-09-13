import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns



""" this is small helper function which concatenates
    a set of consequtive time-lapsed data secordings (sweeps) into 
    single continous sweep with option of down-sampling (for the purpose of illustration)
    Arguments: raw data frame, with > 1 columns, 
        one column ("time" by default) should contain time stamps, 
        if abscent, indices will be used
"""
def read_data():
    data_name = ""
    if len(sys.argv) > 1 :
        file_name = sys.argv[1]       
    else:
        file_name = ""
    if len(file_name) < 2:
        file_name = input(' input file name: ')
    try:

        input_df = pd.read_csv(file_name)
        idx = file_name.find(".")
        if idx > 0 :
            data_name = file_name[:idx]
        else:
            data_name = file_name
        downsample = int(input('enter down-sampling factor: '))
        if downsample < 1:
            downsample = 1
        rows_idx = list(range(0,len(input_df), downsample))
        # print(rows_idx)
        output_df = input_df.iloc[rows_idx,:]
        return output_df, data_name
    except FileNotFoundError :
        return [], data_name



def tmc_arrange(in_df, dataset_name, time_col=None):
    # Y-axis data columns
    y_col_names = [col for col in in_df.columns if col != time_col]
    # parameters for seamless concatenation
    y_tails = in_df[y_col_names].tail(10).mean()
    y_heads = in_df[y_col_names].head(10).mean()
    # y_shift = [0]
    # for i in range(1,len(y_col_names)):
    #     y_shift.append(y_heads.loc[y_col_names[i]] - y_tails.loc[y_col_names[i-1]])
    data_size = len(in_df)
    sweep_dfs = []
    if time_col !=None:
        time_delta = (in_df[time_col].iat[-1] - in_df[time_col].iat[0])*(data_size+1)/data_size
        # print(time_delta)
        sweep_dfs = [in_df.loc[:,[time_col, col]] for col in y_col_names]
        for i in range(1,len(sweep_dfs)):
            sweep_dfs[i][time_col] = sweep_dfs[i][time_col]+ time_delta*i
            y_shift = y_heads.loc[y_col_names[i]] - y_tails.loc[y_col_names[i-1]]
            y_tails.loc[y_col_names[i]] = y_tails.loc[y_col_names[i]] - y_shift
            sweep_dfs[i][y_col_names[i]] = sweep_dfs[i][y_col_names[i]] - y_shift
            sweep_dfs[i].columns=[time_col,y_col_names[0]]
        for df in sweep_dfs:    
            print(df.head())
        out_df = pd.concat(sweep_dfs, ignore_index=True)
        print(out_df.head())
        out_df.to_csv(dataset_name + '_cnt.csv', index=False, mode='w+')
        out_df.plot(x=time_col)
    plt.show()


raw_df, dataset_name = read_data()
print(dataset_name)
tmc_arrange(raw_df, dataset_name, time_col='time')