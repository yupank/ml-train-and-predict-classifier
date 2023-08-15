import pandas as pd
import sys

from src.km_clustering import cluster_analyzer
from src.ml_utils import elbow_cluster_number

def read_input():
    if len(sys.argv) > 1 :
        file_name = sys.argv[1]
    else:
        file_name = ""
    if len(file_name) < 2:
        file_name = input(' input file name: ')
    try:
        file_path = './data/' + file_name
        input_df = pd.read_csv(file_path)
        return input_df
    except FileNotFoundError :
        return []
raw_data = read_input()

if 'time' not in raw_data.columns:
    time_col = None
else:
    time_col = 'time'
print(time_col)
if len(raw_data) > 0:
    print(raw_data.head(5))
else:
    print ("file not found")