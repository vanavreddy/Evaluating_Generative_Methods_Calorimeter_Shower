#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import HighLevelFeatures as HLF
from utils import *
import re


# In[2]:


def file_read(file_name):
    with h5py.File(file_name, "r") as h5f:
        e = h5f['incident_energies'][::].astype(np.float32)  
        shower = h5f['showers'][::].astype(np.float32)
        
    return e, shower


# In[3]:


def extract_name_part(file_name):
    # Use regular expression to extract the desired part of the filename
    match = re.search(r'_([^_]+)\.h5(?:df5)?$', file_name)
    if match:
        return match.group(1)
    else:
        match = re.search(r'_([^_]+)\.hdf5$', file_name)
        return match.group(1)


# In[4]:


path_to_DS1_photon="/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_1/photons"
path_to_DS2_electron="/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2"


# In[5]:


def iterate_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".h5") or filename.endswith(".hdf5"):
            file_path = os.path.join(directory, filename)
            e,shower=file_read(file_path)
            Es.append(e)
            Showers.append(shower)
            name_part = extract_name_part(filename)
            
            if name_part:
                #print("Extracted part from filename:", name_part)
                model_names.append(name_part)


# In[6]:


def _separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()


# In[13]:


def plot_sparsity(HLFs, p_label):
    #print(len(HLFs))
    """ Plot sparsity of relevant layers"""
    g_index=model_names.index('Geant4')
    #print(g_index)
    reference_class=HLFs[g_index]
    for key in HLFs[g_index].GetSparsity().keys():
        lim = (0, 1)
        fig, ax = plt.subplots(2, 1, figsize=(5,4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 20)
        
        counts_ref, bins = np.histogram(1-reference_class.GetSparsity()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color=model_to_color_dict['Geant4'], where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post',
                           color=model_to_color_dict['Geant4'], alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post',
                           color=model_to_color_dict['Geant4'], alpha=0.2 )
        for i in range(len(HLFs)):
            if HLFs[i] == None or g_index==i:
                pass
            else:
                counts, _ = np.histogram(1-HLFs[i].GetSparsity()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(1-HLFs[i].GetSparsity()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=model_names[i], where='post',
                       linewidth=1., alpha=1., color=model_to_color_dict[model_names[i]], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post',
                                   color=model_to_color_dict[model_names[i]], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=model_to_color_dict[model_names[i]], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post', 
                                   color=model_to_color_dict[model_names[i]], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color=model_to_color_dict['Geant4'])
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
        
        ax[1].set_ylabel(r'$\frac{\mathrm{Model}}{\mathrm{GEANT}}$')
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(f"$\\lambda_{{{key}}}$")
        #plt.yscale('log')
        ax[1].set_xlim(*lim)
        ax[0].legend(loc='best', frameon=False, title=p_label, handlelength=1.5, fontsize=15)
        fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
        #if mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(output_dir,
                                'Sparsity_layer_{}_dataset_{}.pdf'.format(key,
                                                                        dataset))
        plt.savefig(filename, format='pdf')
        # if arg.mode in ['all', 'hist-chi', 'hist']:
        # seps = _separation_power(counts_ref, counts_data, bins)
        # print("Separation power of Width Phi layer {} histogram: {}".format(key, seps))
        # with open(os.path.join(output_dir, 'histogram_chi2_{}.txt'.format(dataset)), 'a') as f:
        #     f.write('Sparsity {}: \n'.format(key))
        #     f.write(str(seps))
        #     f.write('\n\n')
        plt.close()


# In[17]:


def plot_Etot_Einc(HLFs, p_label):
    """ plots Etot normalized to Einc histogram """
    
    g_index=model_names.index('Geant4')
    #print(g_index)
    reference_class=HLFs[g_index]

    bins = np.linspace(0.5, 1.5, 31)
    fig, ax = plt.subplots(2,1, figsize=(5, 4.5), gridspec_kw = {"height_ratios": (4,1), "hspace": 0.0}, sharex = True)
        
    counts_ref, bins = np.histogram(reference_class.GetEtot() / reference_class.Einc.squeeze(), bins=bins, density=False)
    counts_ref_norm = counts_ref/counts_ref.sum()
    geant_error = counts_ref_norm/np.sqrt(counts_ref)
    ax[0].step(bins, dup(counts_ref_norm), label='GEANT4', linestyle='-',
                   alpha=0.8, linewidth=1.0, color=model_to_color_dict['Geant4'], where='post')
    ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', 
                       color=model_to_color_dict['Geant4'], alpha=0.2)
    ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', 
                       color=model_to_color_dict['Geant4'], alpha=0.2 )
    for i in range(len(HLFs)):
        if HLFs[i] == None or i==g_index:
            pass
        else:
            counts, _ = np.histogram(HLFs[i].GetEtot() / HLFs[i].Einc.squeeze(), bins=bins, density=False)
            counts_data, bins = np.histogram(HLFs[i].GetEtot() / HLFs[i].Einc.squeeze(), bins=bins, density=False)
            counts_data_norm = counts_data/counts_data.sum()
            ax[0].step(bins, dup(counts_data_norm), label=model_names[i], where='post',
                   linewidth=1., alpha=1., color=model_to_color_dict[model_names[i]], linestyle='-')

            y_ref_err = counts_data_norm/np.sqrt(counts)
            ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post',
                               color=model_to_color_dict[model_names[i]], alpha=0.2)
    
            ratio = counts_data / counts_ref
            ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=model_to_color_dict[model_names[i]], where='post')
            ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post',
                               color=model_to_color_dict[model_names[i]], alpha=0.2)

    ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color=model_to_color_dict['Geant4'])
    ax[1].set_yticks((0.7, 1.0, 1.3))
    ax[1].set_ylim(0.5, 1.5)
    ax[0].set_xlim(bins[0], bins[-1])

    ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
    ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
    
    ax[1].set_xlabel(r'$E_{\mathrm{tot}} / E_{\mathrm{inc}}$')
    ax[0].set_ylabel(r'a.u.')
    ax[1].set_ylabel(r'$\frac{\mathrm{Model}}{\mathrm{GEANT}}$')
    ax[0].legend(loc='best', frameon=False, title=p_label, handlelength=1.5, fontsize=15)
    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
    #if arg.mode in ['all', 'hist-p', 'hist']:
    filename = os.path.join(output_dir, 'Etot_Einc_dataset_{}.pdf'.format(dataset))
    fig.savefig(filename, dpi=300, format='pdf')
    # if arg.mode in ['all', 'hist-chi', 'hist']:
    #     seps = _separation_power(counts_ref, counts_data, bins)
    #     print("Separation power of Etot / Einc histogram: {}".format(seps))
    #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
    #               'a') as f:
    #         f.write('Etot / Einc: \n')
    #         f.write(str(seps))
    #         f.write('\n\n')
    plt.close()


# In[25]:


def plot_ECEtas(list_hlfs,  p_label):
    """ plots center of energy in eta """
    g_index=model_names.index('Geant4')
    #print(g_index)
    reference_class=list_hlfs[g_index]
    for key in reference_class.GetECEtas().keys():
        #print("key: ",key)
        if dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-500., 500.)
        else:
            lim = (-100., 100.)
        fig, ax = plt.subplots(2, 1, figsize=(5, 4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 51)

        counts_ref, bins = np.histogram(reference_class.GetECEtas()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color=model_to_color_dict[model_names[g_index]], where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post',
                           color=model_to_color_dict[model_names[g_index]], alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post',
                           color=model_to_color_dict[model_names[g_index]], alpha=0.2 )
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                counts, _ = np.histogram(list_hlfs[i].GetECEtas()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(list_hlfs[i].GetECEtas()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=model_names[i], where='post',
                       linewidth=1., alpha=1., color=model_to_color_dict[model_names[i]], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post',
                                   color=model_to_color_dict[model_names[i]], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=model_to_color_dict[model_names[i]], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post',
                                   color=model_to_color_dict[model_names[i]], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
 
        #ax[0].set_title(r"Center of Energy in $\Delta\eta$ in layer {}".format(key))
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(f'$\\langle\\eta\\rangle_{{{key}}}$ [mm]')
        ax[0].set_xlim(*lim)
        ax[0].set_yscale('log')
        ax[1].set_ylabel(r'$\frac{\mathrm{Model}}{\mathrm{GEANT}}$')
        ax[0].legend(loc=(0.57, 0.54), frameon=False, title=p_label, handlelength=1.5,  fontsize=15)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

        #if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(output_dir,
                                'ECEta_layer_{}_dataset_{}.pdf'.format(key,
                                                                       dataset))
        plt.savefig(filename, dpi=300, format='pdf')
        # if arg.mode in ['all', 'hist-chi', 'hist']:
        #     seps = _separation_power(counts_ref, counts_data, bins)
        #     print("Separation power of EC Eta layer {} histogram: {}".format(key, seps))
        #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
        #               'a') as f:
        #         f.write('EC Eta layer {}: \n'.format(key))
        #         f.write(str(seps))
        #         f.write('\n\n')
        plt.close()


# In[35]:


def plot_ECPhis(list_hlfs, p_label):
    """ plots center of energy in phi """
    g_index=model_names.index('Geant4')
    #print(g_index)
    reference_class=list_hlfs[g_index]
    for key in reference_class.GetECPhis().keys():
        if dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-500., 500.)
        else:
            lim = (-100., 100.)
        fig, ax = plt.subplots(2, 1, figsize=(5, 4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 51)
        
        counts_ref, bins = np.histogram(reference_class.GetECPhis()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color=model_to_color_dict[model_names[g_index]], where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post',
                           color=model_to_color_dict[model_names[g_index]], alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post',
                           color=model_to_color_dict[model_names[g_index]], alpha=0.2 )
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or i==g_index:
                pass
            else:
                counts, _ = np.histogram(list_hlfs[i].GetECPhis()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(list_hlfs[i].GetECPhis()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=model_names[i], where='post',
                       linewidth=1., alpha=1., color=model_to_color_dict[model_names[i]], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post',color= model_to_color_dict[model_names[i]], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=model_to_color_dict[model_names[i]], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post', 
                                   color=model_to_color_dict[model_names[i]], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
 
        #ax[0].set_title(r"Center of Energy in $\Delta\phi$ in layer {}".format(key))
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(f"$\\langle\\phi\\rangle_{{{key}}}$ [mm]")
        ax[0].set_xlim(*lim)
        ax[0].set_yscale('log')
        ax[1].set_ylabel(r'$\frac{\mathrm{Model}}{\mathrm{GEANT}}$')
        ax[0].legend(loc=(0.57, 0.54), frameon=False, title=p_label, handlelength=1.5,  fontsize=15)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

        #if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(output_dir,
                                'ECPhi_layer_{}_dataset_{}.pdf'.format(key,
                                                                       dataset))
        plt.savefig(filename, dpi=300, format='pdf')
        # if arg.mode in ['all', 'hist-chi', 'hist']:
        #     seps = _separation_power(counts_ref, counts_data, bins)
        #     print("Separation power of EC Phi layer {} histogram: {}".format(key, seps))
        #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
        #               'a') as f:
        #         f.write('EC Phi layer {}: \n'.format(key))
        #         f.write(str(seps))
        #         f.write('\n\n')
        plt.close()


# In[40]:


def plot_ECWidthPhis(list_hlfs,  p_label):
    """ plots width of center of energy in phi """
    g_index=model_names.index('Geant4')
    #print(g_index)
    reference_class=list_hlfs[g_index]
    for key in reference_class.GetWidthPhis().keys():
        if dataset in ['2', '3']:
            lim = (0., 30.)
        elif key in [12, 13]:
            lim = (0., 400.)
        else:
            lim = (0., 100.)
        fig, ax = plt.subplots(2, 1, figsize=(5, 4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 51)
        
        counts_ref, bins = np.histogram(reference_class.GetWidthPhis()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color=model_to_color_dict[model_names[g_index]], where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post',
                           color=model_to_color_dict[model_names[g_index]], alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', 
                           color=model_to_color_dict[model_names[g_index]], alpha=0.2 )
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or i==g_index:
                pass
            else:
                counts, _ = np.histogram(list_hlfs[i].GetWidthPhis()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(list_hlfs[i].GetWidthPhis()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=model_names[i], where='post',
                       linewidth=1., alpha=1., color=model_to_color_dict[model_names[i]], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post',
                                   color=model_to_color_dict[model_names[i]], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, 
                           color=model_to_color_dict[model_names[i]], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post',
                                   color=model_to_color_dict[model_names[i]], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
        
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(r"$\sigma_{\langle\phi\rangle_{" + str(key) + "}}$ [mm]")
        #ax[0].set_title(r"Width of Center of Energy in $\Delta\phi$ in layer {}".format(key))
        ax[0].set_xlim(*lim)
        ax[0].set_yscale('log')
        ax[1].set_ylabel(r'$\frac{\mathrm{Model}}{\mathrm{GEANT}}$')
        ax[0].legend(loc=(0.57, 0.54), frameon=False, title=p_label, handlelength=1.5, fontsize=15)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
 
        #if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(output_dir,
                                'WidthPhi_layer_{}_dataset_{}.pdf'.format(key,
                                                                          dataset))
        plt.savefig(filename, dpi=300, format='pdf')
        # if arg.mode in ['all', 'hist-chi', 'hist']:
        #     seps = _separation_power(counts_ref, counts_data, bins)
        #     print("Separation power of Width Phi layer {} histogram: {}".format(key, seps))
        #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
        #               'a') as f:
        #         f.write('Width Phi layer {}: \n'.format(key))
        #         f.write(str(seps))
        #         f.write('\n\n')
        plt.close()


# In[43]:


def plot_ECWidthEtas(list_hlfs, p_label):
    """ plots width of center of energy in eta """
    g_index=model_names.index('Geant4')
    #print(g_index)
    reference_class=list_hlfs[g_index]
    for key in reference_class.GetWidthEtas().keys():
        if dataset in ['2', '3']:
            lim = (0., 30.)
        elif key in [12, 13]:
            lim = (0., 400.)
        else:
            lim = (0., 100.)
        fig, ax = plt.subplots(2,1, figsize=(5, 4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 51)
        
        counts_ref, bins = np.histogram(reference_class.GetWidthEtas()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color=model_to_color_dict[model_names[g_index]], where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post',
                           color=model_to_color_dict[model_names[g_index]], alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', 
                           color=model_to_color_dict[model_names[g_index]], alpha=0.2 )
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or i==g_index:
                pass
            else:
                counts, _ = np.histogram(list_hlfs[i].GetWidthEtas()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(list_hlfs[i].GetWidthEtas()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=model_names[i], where='post',
                       linewidth=1., alpha=1., color=model_to_color_dict[model_names[i]], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post',
                                   color=model_to_color_dict[model_names[i]], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, 
                           color=model_to_color_dict[model_names[i]], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post',
                                   color=model_to_color_dict[model_names[i]], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
        
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(r"$\sigma_{\langle\eta\rangle_{" + str(key) + "}}$ [mm]")
        #ax[0].set_title(r"Width of Center of Energy in $\Delta\eta$ in layer {}".format(key))
        ax[0].set_xlim(*lim)
        ax[0].set_yscale('log')
        ax[1].set_ylabel(r'$\frac{\mathrm{Model}}{\mathrm{GEANT}}$')
        ax[0].legend(loc=(0.57, 0.54), frameon=False, title=p_label, handlelength=1.5, fontsize=15)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
 
        #if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(output_dir,
                                'WidthEta_layer_{}_dataset_{}.pdf'.format(key,
                                                                          dataset))
        plt.savefig(filename, dpi=300, format='pdf')
        # if arg.mode in ['all', 'hist-chi', 'hist']:
        #     seps = _separation_power(counts_ref, counts_data, bins)
        #     print("Separation power of Width Eta layer {} histogram: {}".format(key, seps))
        #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
        #               'a') as f:
        #         f.write('Width Eta layer {}: \n'.format(key))
        #         f.write(str(seps))
        #         f.write('\n\n')
        plt.close()


# In[14]:


#%%time
Es=[]
Showers=[]
model_names=[]

iterate_files(path_to_DS2_electron)

#change this file path according to your path
#binning_file="/scratch/fa7sa/IJCAI_experiment/homepage/code/binning_dataset_1_photons.xml"
binning_file='/scratch/fa7sa/IJCAI_experiment/homepage/code/binning_dataset_2.xml'
#particle='photon'
particle='electron' #for DS2, DS3
#particle='pion' #for DS1 

HLFs=[]

for i in range(len(model_names)):
    hlf=HLF.HighLevelFeatures(particle,binning_file)
    hlf.Einc=Es[i]
    hlf.CalculateFeatures(Showers[i])
    HLFs.append(hlf)
    

#colors=[]


output_dir="/scratch/fa7sa/Benchmarking_Calorimeter/dataset_2_sparsity"
#dataset='1-photons'
dataset='2'

if dataset == '1-photons':
    p_label = r'$\gamma$ DS-1'
elif dataset == '1-pions':
    p_label = r'$\pi^{+}$ DS-1'
elif dataset == '2':
    p_label = r'$e^{+}$ DS-2'
else:
    p_label = r'$e^{+}$ DS-3'
dup = lambda a: np.append(a, a[-1])

plot_sparsity(HLFs,p_label)


# In[ ]:





# In[18]:


plot_Etot_Einc(HLFs, p_label)


# In[26]:


output_dir="/scratch/fa7sa/Benchmarking_Calorimeter/dataset_2_center_of_energy"
plot_ECEtas(HLFs,p_label)


# In[36]:


output_dir="/scratch/fa7sa/Benchmarking_Calorimeter/dataset_2_center_of_energy_phi"
plot_ECPhis(HLFs,p_label)


# In[44]:


output_dir='/scratch/fa7sa/Benchmarking_Calorimeter/dataset_2_width_etas'
plot_ECWidthEtas(HLFs,p_label)


# In[ ]:




