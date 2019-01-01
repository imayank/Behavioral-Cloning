import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data = pd.read_csv("../data2/data/recording/driving_log.csv")

def expanding_data(data):
     X_center = data.loc[:,'center']
    y_center = data.loc[:,'target']
    X_left = data.loc[:,'left']
    y_left = y_center + 0.1
    X_right = data.loc[:,'right']
    y_right = y_center - 0.1
    
    center_data = pd.concat([X_center,y_center],axis=1,ignore_index=True)
    left_data = pd.concat([X_left,y_left],axis=1,ignore_index=True)
    right_data = pd.concat([X_right,y_right],axis=1,ignore_index=True)
    
    merged_data = pd.concat([center_data,left_data,right_data],axis=0,ignore_index=True)
    merged_data.columns=['path','target']
    
    return merged_data
    

def undersampling(merged_data):
    out = pd.cut(list(merged_data['target']),30,labels=False)
    bins, counts = np.unique(out, return_counts=True)
    avg_counts = np.mean(counts)
    target_counts = int(np.percentile(counts,75))
    indices = np.where(counts>avg_counts)
    target_bins = bins[indices]
    
    target_indices = []
    total_indices = list(range(len(out)))
    remaining_indices = total_indices
    
    for value in target_bins:
        bin_ind = list(np.where(out == value)[0])
        remaining_indices = list(set(remaining_indices) - set(bin_ind))
        random_indices = list(np.random.choice(bin_ind,target_counts, replace=False))
        target_indices.extend(random_indices)
        
    undersampled_indices = np.concatenate([target_indices,remaining_indices])
    undersampled_data = merged_data.loc[undersampled_indices]
    
    return undersampled_data
    
def reset_and_add(undersampled_data):
    undersampled_data = undersampled_data.reset_index()
    undersampled_data["ID"] = list(range(len(undersampled_data)))
