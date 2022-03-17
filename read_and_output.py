import pandas as pd
import numpy as np
import json
import os

def read_dev(root_path, file):
    file_path = os.path.join(root_path, file)
    df = pd.read_csv(file_path, header=0)
    max_epoch = df['F1_all'].idxmax()
    max_value = df['F1_all'].max()
    # print(max_epoch, max_value)
    return max_epoch, max_value

def read_csv(root_path, file, max_epoch, name):
    file_path = os.path.join(root_path, file)
    df = pd.read_csv(file_path, header=0)
    se = df.iloc[max_epoch, :]
    total_epoch,_ = df.shape
    # print(total_epoch)
    # print(se)
    state_dict = {'name':name, 'epoch': se['epoch'], 'total_epoch':total_epoch, 'all':se['F1_all']}
    for i in range(4):
        state_dict['group_%d'%i] = se['F1_group:%d'%i]
    state_dict['worst'] = se['F1_wg']
    return state_dict
    

def read_results(root_path):
    files = os.listdir(root_path)
    max_epoch, max_value = read_dev(root_path, 'val_eval.csv')
    state_list = []
    for file in files:
        if "_eval.csv" in file and "test_" in file:
            if file == 'test_eval.csv':
                continue 
            name = file.split('_')[1]
            # print(name)
            state_dict = read_csv(root_path, file, max_epoch, name)
            # print(state_dict)
            state_list.append(state_dict)
    return state_list

def read_dirs(root_path):
    files = sorted(os.listdir(root_path))
    for file in files:
        file_path = os.path.join(root_path, file)
        if "0119_" not in file: continue
        # if "france" in file: continue
        if os.path.isdir(file_path):
            print(file_path)
            country = file.split('_')[1]
            # print(country)
            try:
                state_list = read_results(file_path)
            except pd.errors.EmptyDataError as m:
                print("None\n")
                continue
            
            best_epoch, total_epoch = 0, 0
            avg_score , worst_score = 0, 0
            zs_avg_score, zs_worst_score = 0, 0
            for res in state_list:
                if res['name'] == country:
                    avg_score, worst_score = res['all'], res['worst']
                    best_epoch, total_epoch = res['epoch'], res['total_epoch']

                else:
                    zs_avg_score += res['all']
                    zs_worst_score += res['worst']
            zs_avg_score /= 4
            zs_worst_score /= 4
            print("%s [%d/%d] %.2f/%.2f (%.2f/%.2f)\n"%(country, best_epoch, total_epoch, avg_score*100, worst_score*100, zs_avg_score*100, zs_worst_score*100))

            
            

if __name__ == "__main__":
    read_dirs('logs')