#!/usr/bin/env python
# coding: utf-8


import numpy as np
import h5py
import matplotlib.pyplot as plt
import HighLevelFeatures as HLF
from pathlib import Path
import configargparse
import sys
import pandas as pd


def file_read(file_name):
    with h5py.File(file_name, "r") as h5f:
        energies = h5f['incident_energies'][::].astype(np.float32)  
        showers = h5f['showers'][::].astype(np.float32)
        
    return energies, showers


def calculate_separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()



#min_energy=0.5e-6/0.033
min_energy=0.5e-4/0.033
#min_energy=0.1/0.033
x_scale='log'
model_to_color = {'Geant4': 'black',
                  'CaloDiffusion': 'salmon',
                  'CaloScore': 'Blue',
                  'CaloINN': 'Green'}
def plot_E_group_layers(ref_model, hlf_classes, model_names, plot_filename, e_range, dataset_num):
    """ plots energy deposited in 5 consecutive layers by creating a group of 5 layers"""
    # this is only applicable for dataset 2 and dataset 3. Dataset 1 does not need this

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    
    keys = [[i+j for j in range(5)] for i in range(0, 45, 5)]
    axs = axs.flatten()

    print("Range: ", e_range)

    sep_powers_all = []
    layers_all = []

    for i, key in enumerate(keys):
        
        sep_powers = []
        layers = []

        ref_shape = hlf_classes[ref_model].GetElayers()[0].shape[0]
        ref_selected = [hlf_classes[ref_model].GetElayers()[i].reshape(ref_shape, 1)/1000 for i in key]#turning into GeV
        ref_combined = np.concatenate(ref_selected, axis=1)
        #ref_mean = np.mean(ref_combined, axis=1, keepdims=True) 
        ref_mean = np.sum(ref_combined, axis=1, keepdims=True) 

        if x_scale == 'log':
            bins = np.logspace(np.log10(min_energy),
                               np.log10(ref_mean.max()), 40)
        else:
            bins = 40

        ref_counts, bins, _ = axs[i].hist(ref_mean, bins=bins, label=ref_model, density=True, 
                            histtype='step',color=model_to_color[ref_model], alpha=0.2, linewidth=3.)

        this_layer = str(key[0]) + "-" + str(key[4]) 
        sep_powers.append(this_layer)
        sep_powers.append(0) # seperation power baseline, Geant4 with itself

        legend_names = ['Geant4']
            
        for j, model in enumerate(model_names):
            if model != ref_model:

                model_shape = hlf_classes[model].GetElayers()[0].shape[0]
                selected_hlf = [hlf_classes[model].GetElayers()[i].reshape(model_shape, 1)/1000 for i in key]#turning into GeV
                combined_hlf = np.concatenate(selected_hlf, axis=1)
                #hlf_means = np.mean(combined_hlf, axis=1, keepdims=True) 
                hlf_means = np.sum(combined_hlf, axis=1, keepdims=True) 
                model_counts, _, _ = axs[i].hist(hlf_means, label=model, bins=bins,
                                    histtype='step', color=model_to_color[model], linewidth=3., alpha=1., density=True)
            
                axs[i].set_title("layer {} to {}".format(key[0],key[4]))
                axs[i].set_xlabel(r'$E$ [GeV]')
                axs[i].set_yscale('log')
                axs[i].set_xscale('log')
            
                sep_power = calculate_separation_power(ref_counts, model_counts, bins)
                sep_powers.append(sep_power)

                legend_names.append(model)
           
        layers.append(str(key[0]) + "-" + str(key[4]))

        sep_powers_all.append(sep_powers)
        #print("separation power {} is {}, for layers {}".format(model_names, sep_powers, this_layer))

    # Plot E layers
    fig.legend(legend_names, fontsize=12, prop={'size': 10},
            loc='upper center', bbox_to_anchor=(0.5, 1.0), ncols=4)
    plt.tight_layout(pad=3.0)
    
    plt.savefig(plot_filename, dpi=300)

    # get the colors matching the order in legend_names
    colors = []
    for model in legend_names:
        colors.append(model_to_color[model])

    # Plot chi-squared a.k.a seperation power 
    legend_names.insert(0, 'Layers')
    df = pd.DataFrame(sep_powers_all, columns=legend_names)
    df.plot(x='Layers', y=legend_names[1:], color=colors, figsize=(10,5))
    # Set the x-axis label
    plt.xlabel('Layers')
    # Set the y-axis label
    plt.ylabel('Separation power')
    plt.legend(ncol=4, bbox_to_anchor=(0.5, 0.98),
           loc='upper center', fontsize=12)
    # save plot
    plt.savefig("sep_pow_ds" + str(dataset_num) + "_" + e_range + ".pdf", bbox_inches='tight', dpi=300)

    plt.close()

def extract_model_names(evaluate_files_list):
    model_names = []
    for name in evaluate_files_list:
        m_name = name.split('/')[-1].split('.')[0].split('_')[-1]
        model_names.append(m_name)

    return model_names
    
  
if __name__ == "__main__":
    
    parser = configargparse.ArgumentParser(default_config_files=[])
    parser.add_argument('--dataset_path', type=str, required=True, 
                        default='/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2',
                        help='path to generated h5/hdf5 files are stored')
    parser.add_argument('--binning_file', type=str, required=True, 
                        default='/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/binning_files/binning_dataset_2.xml',
                        help='path to binning file')
    parser.add_argument('--particle_type', type=str, required=True,
                        help='type of the particle being evaluated e.g., photon, pion, electron')
    parser.add_argument('--dataset_num', type=int, required=True, default=2,
                        help='dataset number e.g., 1, 2, 3')
    
    args = parser.parse_args()
    binning_file= args.binning_file
    particle = args.particle_type
    target_energies = 10**np.linspace(3, 6, 4)
    #target_energies = [10e6, 11e6]#, 50e6, 55e6, 100e6, 110e6]
    
    evaluate_path = args.dataset_path
    evaluate_files_list = [str(p) for p in Path(evaluate_path).rglob('*.h5')]
   
    model_names = extract_model_names(evaluate_files_list)
    print("Extracted Model Names: ", model_names)
   
    hlfs = dict()
    for model in model_names:
        hlfs[model] = HLF.HighLevelFeatures(particle, binning_file)

    # plot energy ranges seperated
    for i in range(len(target_energies)-1):
        for idx,eval_model in enumerate(evaluate_files_list):

            energies, showers = file_read(eval_model)
            model = model_names[idx]
            
            shower_bins = ((energies >= target_energies[i]) & \
                             (energies < target_energies[i+1])).squeeze()
            hlfs[model].Einc = energies[shower_bins]
            hlfs[model].CalculateFeatures(showers[shower_bins])
            
        e_range = str(int(target_energies[i]/1000))+'GeV_'+str(int(target_energies[i+1]/1000))+'GeV'
        plot_filename =  'E_layers_dataset_{}_{}.pdf'.format(args.dataset_num, e_range)
        ref = 'Geant4'
        plot_E_group_layers(ref, hlfs, model_names, plot_filename, e_range, args.dataset_num)

        model_names = extract_model_names(evaluate_files_list)


    # plot all energy ranges
    hlfs = dict()
    for model in model_names:
        hlfs[model] = HLF.HighLevelFeatures(particle, binning_file)

    for idx,eval_model in enumerate(evaluate_files_list):
        energies, showers = file_read(eval_model)
        model = model_names[idx]

        hlfs[model].Einc = energies
        hlfs[model].CalculateFeatures(showers)

    plot_filename =  'E_layers_dataset_{}_{}.pdf'.format(args.dataset_num, 'all')
    ref = 'Geant4'
    e_range = 'all'
    plot_E_group_layers(ref, hlfs, model_names, plot_filename, e_range, args.dataset_num)

    print("done plotting...")

