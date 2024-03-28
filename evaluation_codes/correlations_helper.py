import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import torch
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


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
        
        
def calculateCorrelation(layer_real):
    dim=layer_real.shape[1]
    correlation_matrix = np.ones((dim, dim))
    p_value_matrix = np.zeros((dim, dim))

    # Loop through each pair of layers and compute correlations and p-values
    for i in range(dim):
        for j in range(dim):
            if i != j:  # Exclude self-correlation (diagonal elements)
                corr, p_value = pearsonr(layer_real[:, i], layer_real[:, j])
                correlation_matrix[i, j] = corr
                p_value_matrix[i, j] = p_value
                
                
    return correlation_matrix,p_value_matrix


def draw_heatmap(correlation_matrix,name,data_name):
    sns.set()  # Set seaborn style
    plt.figure(figsize=(10, 8))  # Set the figure size

    # Create the heatmap
    #sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f',linecolor='white',linewidths=0.3)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    #cbar_kws={'ticks': [0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 1.0]}
    # Set labels and title
    plt.xlabel('Layers')
    plt.ylabel('Layers')
    plt.title(f'Correlation Heatmap between Layers of {data_name}')
    plt.gca().invert_yaxis()
    plt.savefig(name, bbox_inches='tight')  # Save the figure


    # Show the heatmap
    plt.show()
    
    
def draw_heatmap_dual(correlation_matrix1, correlation_matrix2, name, data_name1,data_name2,custom_xticks):
    sns.set()  # Set seaborn style
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))  # Create a figure with two subplots
    print(correlation_matrix1.shape,correlation_matrix2.shape)
    

    # Create the heatmap for the first matrix
    overall_min = min(correlation_matrix1.min(), correlation_matrix2.min())
    overall_max = max(correlation_matrix1.max(), correlation_matrix2.max())

    sns.heatmap(correlation_matrix1, annot=False, cmap='coolwarm', cbar_kws={'ticks': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},fmt='.2f', linecolor='white', linewidths=0.5, ax=axs[0],vmin=overall_min, vmax=overall_max)
    axs[0].set_xlabel('Layers')
    axs[0].set_ylabel('Layers')
    axs[0].set_title(f'Correlation Heatmap - {data_name1}')
    axs[0].invert_yaxis()
    #axs[0].set_xticks(custom_xticks)  # Set custom x-axis ticks
    axs[0].set_xticklabels(custom_xticks) 
    # Create the heatmap for the second matrix
    sns.heatmap(correlation_matrix2, annot=False, cmap='coolwarm',cbar_kws={'ticks': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}, fmt='.2f', linecolor='white', linewidths=0.5, ax=axs[1],vmin=overall_min, vmax=overall_max)
    axs[1].set_xlabel('Layers')
    axs[1].set_ylabel('Layers')
    axs[1].set_title(f'Correlation Heatmap - {data_name2}')
    axs[1].invert_yaxis()
    #axs[1].set_xticks(custom_xticks)  # Set custom x-axis ticks
    axs[1].set_xticklabels(custom_xticks) 
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.15, right=0.85)

    plt.savefig(name, bbox_inches='tight')  # Save the figure

    # Show the heatmap
    plt.show()
    
    
def barOnbar(gen,real,dataset,num,fname):
    plt.figure(figsize=(10, 6))
    x = np.arange(45)
    for i in range(45):
        plt.bar(x[i], gen[i], color='blue', alpha=0.7)  # Blue bars for array 'a'
        plt.bar(x[i], real[i], color='yellow', alpha=0.7)   # Red bars for array 'b'

    # Customizing the plot
    plt.xlabel('Layer Number')
    plt.ylabel('Correlation Values')
    plt.title(f'Histogram-like Plot of CaloDiff and Geant4 for {dataset} in Layer {num}')
    plt.legend(['Calodiff', 'Geant4'])
    plt.xticks(np.arange(0, 45, 5))
    plt.savefig(fname)

    # Show the plot
    plt.show()
    
    
def barBybar(gen,real,dataset,num):
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(45)
    for i in range(45):
        plt.bar(x[i]-bar_width/2, gen[i],width=bar_width, color='blue', alpha=0.5)  # Blue bars for array 'a'
        plt.bar(x[i]+bar_width/2, real[i], width=bar_width, color='red', alpha=0.5)   # Red bars for array 'b'

    # Customizing the plot
    plt.xlabel('Layer Number')
    plt.ylabel('Correlation Values')
    plt.title(f'Histogram-like Plot of CaloDiff and Geant4 for {dataset} in Layer {num}')
    plt.legend(['Calodiff', 'Geant4'])
    plt.xticks(np.arange(0, 45, 5))
    plt.savefig('side_by_side_bars.png')

    # Show the plot
    plt.show()


