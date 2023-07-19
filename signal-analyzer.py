from models.ml_cluster_analyser_2 import cluster_analyzer
import pandas as pd
import sys
def read_input():
    for arg in sys.argv:
        print(f'sys arg: {arg}')
    try:
        file_path = './data/' + input(' input file name: ')
        input_df = pd.read_csv(file_path)
        return input_df
    except FileNotFoundError :
        return []
red_data = read_input()
if len(red_data) > 0:
    print(red_data.head(5))
else:
    print ("file not found")