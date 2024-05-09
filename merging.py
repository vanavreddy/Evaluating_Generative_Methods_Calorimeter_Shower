import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import HighLevelFeatures as HLF

def file_read(file_name):
    with h5py.File(file_name, "r") as h5f:
        e = h5f['incident_energies'][::].astype(np.float32)  
        shower = h5f['showers'][::].astype(np.float32)
        
    return e, shower


def merge_hdf5_files(file1_path, file2_path, output_path):
    e1, shower1 = file_read(file1_path)
    e2, shower2 = file_read(file2_path)
    
    # Create a new HDF5 file for the merged data
    with h5py.File(output_path, 'w') as output_file:
        output_file.create_dataset('incident_energies', data=np.concatenate((e1, e2)))
        output_file.create_dataset('showers', data=np.concatenate((shower1, shower2)))
        
file_ds3_1='/scratch/fa7sa/IJCAI_experiment/dataset_3/dataset_3_3.hdf5'
file_ds3_2='/scratch/fa7sa/IJCAI_experiment/dataset_3/dataset_3_4.hdf5'

merge_hdf5_files(file_ds3_1,file_ds3_2,'dataset_3_electron_Geant4.h5')