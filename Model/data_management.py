import numpy as np
import pandas as pd
import os

def get_file_list_in_one_folder(folderPath):
    file_array = []
    files = os.listdir(folderPath)
    for file in files:
        if file != ".DS_Store":
            file_array.append(file)
    return file_array

def get_data_per_file(path):
    dataset={}
    files = get_file_list_in_one_folder(path)
    for file in files:
        file_name = file.split('.')
        if len(file_name) > 1 and file_name[len(file_name)-1] == 'csv':
            data = pd.read_csv(path + '/' + file)
            data_str = data.to_numpy()
            data_int = data_str.astype(int)
            dataset.update({
                file_name[0]:data_int
            })
    return dataset