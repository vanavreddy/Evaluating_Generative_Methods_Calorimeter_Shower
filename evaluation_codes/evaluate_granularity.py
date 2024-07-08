import sys
import os
import getpass
import torch.optim as optim
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
torch.manual_seed(32)
import numpy as np
np.random.seed(32)
import matplotlib.pyplot as plt
import h5py

# takes path to .h5/h5py files and returns tensor of samples
def read_sample_files(path):
    samples = h5py.File(path)
    shower = samples['showers'][:]
    energy = samples['incident_energies'][:]
    return torch.tensor(shower)

def measure_granularity(data_tensor, shift_index = 0):
    segment_length = 144  # Length of each segment
    num_segments = 45     # Number of segments per sample
    num_samples = data_tensor.size(0)  # Number of samples in the data tensor

    shifts = torch.full((num_samples,), shift_index, dtype=torch.int64, device=data_tensor.device).unsqueeze(-1).expand(-1, num_segments)

    # The result is a tensor of shape (num_samples, num_segments, segment_length)
    segments = data_tensor.unfold(1, segment_length, segment_length)

    # Create an indices tensor of shape (num_samples, num_segments, segment_length)
    indices = torch.arange(segment_length, device=data_tensor.device).repeat(num_samples, num_segments, 1)
    # Adjust the indices by adding the shifts and applying modulo operation to wrap around
    indices = (indices + shifts.unsqueeze(-1)) % segment_length  # Ensure correct broadcasting

    # Gather elements from the segments tensor using the adjusted indices
    rotated_segments = torch.gather(segments, 2, indices)

    # Reshape the rotated_segments tensor back to the original shape of data_tensor
    result_tensor = rotated_segments.view(num_samples, -1)

    # Compute the difference between the original data tensor and the result tensor
    diffs = data_tensor - result_tensor

    return diffs

def measure_stochastic_granularity(data_tensor):
    segment_length = 144  # Length of each segment
    num_segments = 45     # Number of segments per sample
    num_samples = data_tensor.size(0)  # Number of samples in the data tensor

    # Use PyTorch to generate a random integer array of shape (num_samples,) with values between 0 and 15
    random_array = torch.randint(0, 16, (num_samples,), dtype=torch.int64, device=data_tensor.device)

    # Multiply random_array by 9 and expand it to shape (num_samples, num_segments)
    shifts = (random_array * 9).unsqueeze(-1).expand(-1, num_segments)

    # Unfold the data tensor to create segments of length 144
    # The result is a tensor of shape (num_samples, num_segments, segment_length)
    segments = data_tensor.unfold(1, segment_length, segment_length)

    # Create an indices tensor of shape (num_samples, num_segments, segment_length)
    indices = torch.arange(segment_length, device=data_tensor.device).repeat(num_samples, num_segments, 1)
    # Adjust the indices by adding the shifts and applying modulo operation to wrap around
    indices = (indices + shifts.unsqueeze(-1)) % segment_length  # Ensure correct broadcasting

    # Gather elements from the segments tensor using the adjusted indices
    rotated_segments = torch.gather(segments, 2, indices)

    # Reshape the rotated_segments tensor back to the original shape of data_tensor
    result_tensor = rotated_segments.view(num_samples, -1)

    # Compute the difference between the original data tensor and the result tensor
    diffs = data_tensor - result_tensor

    return diffs



def plot_granularity(granularity_reference, granularity_model, samples_reference, samples_model, model_name):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    indices = range(1000, 10001, 1000)

    for ax, i in zip(axes.flatten(), indices):
        ax.plot(granularity_reference[i], label='Geant4', alpha=0.5)
        ax.plot(granularity_model[i], label=model_name, alpha=0.5)
        total_energy = int((torch.sum(samples_reference[i])+torch.sum(samples_model[i]))/2)
        ax.set_xlabel('Voxel ID')
        ax.set_ylabel('abs(Diffs)')
        ax.legend()
        ax.set_title(f'Granularity Map for Event {i} \n Input Energy = {total_energy} MeV')

    plt.savefig("granularity_"+model_name+".png", bbox_inches='tight', dpi=300)


def plot_granularity_histogram(reference_std, model_std, model_name):
    combined_data = np.concatenate((reference_std, model_std))
    bins = np.histogram_bin_edges(combined_data, bins=400)
    plt.hist(reference_std, log=True, color = 'red', bins=bins,alpha = 0.4, label = "Geant4", density = True)
    plt.hist(model_std, log=True, color = 'blue', alpha = 0.4, bins=bins, label = model_name, density = True)
    plt.xlim([0, 200])
    plt.xlabel('Standard Deviation')
    plt.ylabel('Density')
    plt.legend()
    plt.title("Granularity STD Histogram")
    plt.tight_layout()
    plt.savefig("Granularity_STD_histogram_"+model_name+".png", bbox_inches='tight', dpi=300)
    plt.close()

def measure_single_granularity(data_tensor_1, index):
    """
    Measure the granularity of a calorimeter image given as a 1D array using PyTorch.
    
    Parameters:
    data (array-like): 1D array representing the calorimeter image.
    
    Returns:
    float: A measure of the granularity.
    """
    # Calculate the differences between consecutive elements
    #     diffs_1 = (data_tensor_1[:,index:] - data_tensor_1[:,:-index])/torch.mean(data_tensor_1[:,:-index], dim = 1, keepdim=True)
    diffs_1 = (data_tensor_1[:,index:] - data_tensor_1[:,:-index])
    #     std_g = torch.std(diffs_1, dim = 1)
    return diffs_1


def plot_shifting_index(geant4_samples, calodiff_samples, caloscore_samples, caloinn_samples):
    index = []
    error_geant4_list = []
    error_calodiff_list = []
    error_caloscore_list = []
    error_caloinn_list = []

    ratio_list_calodiff = []
    ratio_list_caloscore = []
    ratio_list_caloinn = []

    for i in range(1, 30):
        error_geant4 = torch.mean(torch.std(measure_single_granularity(geant4_samples, i),dim=0))
        error_calodiff = torch.mean(torch.std(measure_single_granularity(calodiff_samples, i),dim=0))
        error_caloscore = torch.mean(torch.std(measure_single_granularity(caloscore_samples, i),dim=0))
        error_caloinn = torch.mean(torch.std(measure_single_granularity(caloinn_samples, i),dim=0))

        ratio_calodiff = error_geant4/error_calodiff
        ratio_caloscore = error_geant4/error_caloscore
        ratio_caloinn = error_geant4/error_caloinn

        error_geant4_list.append(error_geant4)
        error_calodiff_list.append(error_calodiff)
        error_caloscore_list.append(error_caloscore)
        error_caloinn_list.append(error_caloinn)

        ratio_list_calodiff.append(ratio_calodiff)
        ratio_list_caloscore.append(ratio_caloscore)
        ratio_list_caloinn.append(ratio_caloinn)
        index.append(i)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(index, error_geant4_list, label='Geant4', color='black')
    ax1.plot(index, error_calodiff_list, label='CaloDiffusion', color='red')
    ax1.plot(index, error_caloscore_list, label='CaloScore', color='green')
    ax1.plot(index, error_caloinn_list, label='CaloINN', color='blue')
    ax1.legend()
    ax1.set_ylabel('Mean STD')

    ax2.plot(index, ratio_list_calodiff, label='Ratio Geant4/CaloDiffusion', color='red')
    ax2.plot(index, ratio_list_caloscore, label='Ratio Geant4/CaloScore', color='green')
    ax2.plot(index, ratio_list_caloinn, label='Ratio Geant4/CaloINN', color='blue')
    ax2.legend()
    ax2.set_xlabel('Shifting Index')
    ax2.set_ylabel('Ratio')

    plt.savefig("shifting_idx.png", bbox_inches='tight', dpi=300)


geant4_file = '/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/dataset_2_electron_Geant4.h5'
calodiff_file = '/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/dataset_2_electron_CaloDiffusion.h5'
caloscore_file = '/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/dataset_2_electron_CaloScore.h5'
caloinn_file = '/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/dataset_2_electron_CaloINN.h5'

geant4_samples = read_sample_files(geant4_file)
calodiff_samples = read_sample_files(calodiff_file)
caloscore_samples = read_sample_files(caloscore_file)
caloinn_samples = read_sample_files(caloinn_file)

geant4_gran = measure_stochastic_granularity(geant4_samples)
calodiff_gran = measure_stochastic_granularity(calodiff_samples)
caloscore_gran = measure_stochastic_granularity(caloscore_samples)
caloinn_gran = measure_stochastic_granularity(caloinn_samples)

'''
plot_granularity(geant4_gran, calodiff_gran, geant4_samples, calodiff_samples, model_name='CaloDiffusion')
plot_granularity(geant4_gran, caloscore_gran, geant4_samples, caloscore_samples, model_name='CaloScore')
plot_granularity(geant4_gran, caloinn_gran, geant4_samples, caloinn_samples, model_name='CaloINN')
'''

std_geant4 = torch.std(geant4_gran, dim=1)
std_calodiff = torch.std(calodiff_gran, dim=1)
std_caloscore = torch.std(caloscore_gran, dim=1)
std_caloinn = torch.std(caloinn_gran, dim=1)

plot_granularity_histogram(std_geant4, std_calodiff, 'CaloDiffusion')
plot_granularity_histogram(std_geant4, std_caloscore, 'CaloScore')
plot_granularity_histogram(std_geant4, std_caloinn, 'CaloINN')

#plot_shifting_index(geant4_samples, calodiff_samples, caloscore_samples, caloinn_samples)
