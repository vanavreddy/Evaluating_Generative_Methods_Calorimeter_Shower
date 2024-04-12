#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import HighLevelFeatures as HLF

import re


# In[3]:


def file_read(file_name):
    with h5py.File(file_name, "r") as h5f:
        e = h5f['incident_energies'][::].astype(np.float32)  
        shower = h5f['showers'][::].astype(np.float32)
        
    return e, shower


# In[4]:


def extract_name_part(file_name):
    # Use regular expression to extract the desired part of the filename
    match = re.search(r'_([^_]+)\.h5(?:df5)?$', file_name)
    if match:
        return match.group(1)
    else:
        match = re.search(r'_([^_]+)\.hdf5$', file_name)
        return match.group(1)


# ## We have separate directory for each dataset. In the directory we store our generated samples from different models and Geant4. The name pattern of the file is like this 'dataset_n_particleName_x.h5' where n denotes the dataset number, particleName can be photons, pions and electron. x denotes the model Name.

# In[5]:


#change this path according to your need
path_to_DS1_photon='/home/vv3xu/Calorimeter_Data_To_Evaluate/dataset_3'


# In[6]:


def iterate_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".h5") or filename.endswith(".hdf5"):
            file_path = os.path.join(directory, filename)
            e,shower=file_read(file_path)
            Es.append(e)
            Showers.append(shower)
            name_part = extract_name_part(filename)
            
            if name_part:
                print("Extracted part from filename:", name_part)
                model_names.append(name_part)


# In[ ]:


Es=[]
Showers=[]
model_names=[]

iterate_files(path_to_DS1_photon)


# In[ ]:


model_names


# In[ ]:


def _separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()


# In[ ]:


#change this file path according to your path
binning_file="/project/biocomplexity/calorimeter/CaloDiffusionPaper/CaloChallenge/CaloChallenge/code/binning_dataset_3.xml"
#particle='photon'
particle='electron'
HLFs=[]


# In[ ]:


for i in range(len(model_names)):
    hlf=HLF.HighLevelFeatures(particle,binning_file)
    hlf.Einc=Es[i]
    hlf.CalculateFeatures(Showers[i])
    HLFs.append(hlf)


# In[ ]:


HLFs[0].GetElayers().keys()


# In[ ]:


colors=['black','blue','green','salmon','orange','yellow']
linestyles=['dashed','solid','dotted','dashdot']


# # plot_E_layers is only called for dataset 1. It shows Layer wise energy distribution

# In[ ]:


x_scale='log'
min_energy=10

def plot_E_layers(hlf_classes, x_scale,model_names,min_energy):
    for key in hlf_classes[0].GetElayers().keys():
        plt.figure(figsize=(6, 6))
        
        if x_scale == 'log':
            bins = np.logspace(np.log10(min_energy),
                               np.log10(hlf_classes[0].GetElayers()[key].max()),
                               40)
        else:
            bins = 40
            
        hists=[]
            
            
        for i  in range(len(hlf_classes)):
            counts_data, _, _ = plt.hist(hlf_classes[i].GetElayers()[key], label=model_names[i], bins=bins, color=colors[i],
                                         histtype='step', linewidth=3., alpha=1., density=True, linestyle=linestyles[i])
            hists.append(counts_data)
            
        plt.title("Energy deposited in layer {}".format(key))
        plt.xlabel(r'$E$ [MeV]')
        plt.yscale('log')
        if x_scale=='log':
            plt.xscale('log')
        plt.legend(fontsize=12,loc='upper right')
        plt.tight_layout(pad=3.0)
        
        filename = 'E_layer_{}_dataset_{}.pdf'.format(
            key,
            '1-photons')
        
        plt.savefig(filename)
        
        try:
            gi = model_names.index('Geant4')
            #print("Index of 'Geant4':", gi)
        except ValueError:
            print("'Geant4' not found in the list.")
            
        seps=[]
        #print(hists[gi])
        for i in range(len(hists)):
            #if gi != i:
            sep=_separation_power(hists[gi],hists[i],bins)
            seps.append(sep)
                
        with open('histogram_chi2_{}.txt'.format('1-photons'), 'a') as f:
            f.write('E layer {}: \n'.format(key))
            for i in range(len(model_names)):
                f.write('for {}: {} \n'.format(model_names[i], seps[i]))
            f.write('\n\n')
            
            
        plt.close()
    print("Plot generation completed..")


# In[ ]:


plot_E_layers(HLFs,x_scale,model_names,min_energy)


# 

# In[ ]:




