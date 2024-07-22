import numpy as np
import h5py
import matplotlib.pyplot as plt
import os


def file_read(file_name):
    with h5py.File(file_name, "r") as h5f:
        e = h5f['incident_energies'][::].astype(np.float32)  
        shower = h5f['showers'][::].astype(np.float32)
        
    return e, shower


def mean_energy(xy, data):
    if xy=='r':
        axis=(1,2)
    if xy=='a':
        axis=(1,3)
    if xy=='z':
        axis=(2,3)
    
    result = np.sum(data, axis=axis)/1000 # GeV
    mean_result = np.mean(result, axis=0)
    
    return mean_result
    
    
def plot_mean_energy(xy,data,dataset,labels):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    m=['o','x','s','d']
    ls=['-','--','-.',':']
    
    for i in range(len(data)):
        ax1.plot(data[i], marker=m[i], linestyle=ls[i], label=labels[i])
    

        # Set x-axis ticks
        if xy=='a':
            ax1.set_xticks(np.arange(16))
            ax1.set_xticklabels(np.arange(1, 17))
            ax2.set_xticks(np.arange(16))
            ax2.set_xticklabels(np.arange(1, 17))
            ax1.set_xlabel('Angular-bins')
            ax2.set_xlabel('Angular-bins')
           
        if xy=='r':
            ax1.set_xticks(np.arange(9))
            ax1.set_xticklabels(np.arange(1, 10))
            ax2.set_xticks(np.arange(9))
            ax2.set_xticklabels(np.arange(1, 10))
            
            ax1.set_xlabel('Radial-bins')
            ax2.set_xlabel('Radial-bins')
            

        if xy=='z':
            ax1.set_xticks(np.arange(0,45,5))
            ax1.set_xticklabels(np.arange(1, 46,5))
            ax2.set_xticks(np.arange(0,45,5))
            ax2.set_xticklabels(np.arange(1, 46,5))
            
            ax1.set_xlabel('Layers')
            ax2.set_xlabel('Layers')
            
    if dataset=='3':
        # Set x-axis ticks
        if xy=='a':
            ax1.set_xticks(np.arange(0,50,5))
            ax1.set_xticklabels(np.arange(1, 51,5))
            ax2.set_xticks(np.arange(0,50,5))
            ax2.set_xticklabels(np.arange(1, 51,5))
            ax1.set_xlabel('Angular-bins')
            ax2.set_xlabel('Angular-bins')
            

        if xy=='r':
            ax1.set_xticks(np.arange(18))
            ax1.set_xticklabels(np.arange(1, 19))
            ax2.set_xticks(np.arange(18))
            ax2.set_xticklabels(np.arange(1, 19))
            ax1.set_xlabel('Radial-bins')
            ax2.set_xlabel('Radial-bins')
          

        if xy=='z':
            ax1.set_xticks(np.arange(0,45,5))
            ax1.set_xticklabels(np.arange(1, 46,5))
            ax2.set_xticks(np.arange(0,45,5))
            ax2.set_xticklabels(np.arange(1, 46,5))
            ax1.set_xlabel('Layers')
            ax2.set_xlabel('Layers')
            
    ax1.set_ylabel('Mean Energy[GeV]')  
    ax2.set_ylabel('Percentage Difference')
    ax1.legend()
    
    
    plt.title(f'Dataset {dataset}')
    
    colors = {'CaloDiffusion': 'orange', 'CaloINN': 'green', 'CaloScore': 'red'}
    for i in range(1,len(data)):
        ratio = 100 * np.divide(data[i] - data[0], data[0])
        ax2.plot(ratio, color=colors[labels[i]], linewidth=2, linestyle=ls[i], label=labels[i])

    ax2.legend()   
    plt.savefig(f'mean_energy_{xy}_dataset_{dataset}.png')
    # Show the plot
    plt.tight_layout()
    plt.show()  
    
    
    
    
def prep_dataset(ds):
    g_path=f'/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_{ds}/dataset_{ds}_electron_Geant4.h5'
    d_path=f'/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_{ds}/dataset_{ds}_electron_CaloDiffusion.h5'
    i_path=f'/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_{ds}/dataset_{ds}_electron_CaloINN.h5'
    s_path=f'/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_{ds}/dataset_{ds}_electron_CaloScore.h5'
    e_g,shower_g=file_read(g_path)
    e_d,shower_d=file_read(d_path)
    e_i,shower_i=file_read(i_path)
    e_s,shower_s=file_read(s_path)
    l=45
    if ds=='2':
        shape=[-1,45,16,9]
        a=16
        r=9
        
    if ds=='3':
        shape=[-1,45,50,18]
        a=50
        r=18
        
    shower_g = shower_g.reshape(shape)
    shower_d=shower_d.reshape(shape)
    shower_i=shower_i.reshape(shape)
    shower_s=shower_s.reshape(shape)
    
    alpha_mean_g=mean_energy('a',shower_g).reshape([a,1]).flatten()
    alpha_mean_s=mean_energy('a',shower_s).reshape([a,1]).flatten()
    alpha_mean_d=mean_energy('a',shower_d).reshape([a,1]).flatten()
    alpha_mean_i=mean_energy('a',shower_i).reshape([a,1]).flatten()
    data_a=[alpha_mean_g,alpha_mean_d,alpha_mean_i,alpha_mean_s]
    
    r_mean_g=mean_energy('r',shower_g).reshape([r,1]).flatten()
    r_mean_d=mean_energy('r',shower_d).reshape([r,1]).flatten()
    r_mean_i=mean_energy('r',shower_i).reshape([r,1]).flatten()
    r_mean_s=mean_energy('r',shower_s).reshape([r,1]).flatten()
    data_r=[r_mean_g,r_mean_d,r_mean_i,r_mean_s]
    
    l_mean_g=mean_energy('z',shower_g).reshape([l,1]).flatten()
    l_mean_d=mean_energy('z',shower_d).reshape([l,1]).flatten()
    l_mean_i=mean_energy('z',shower_i).reshape([l,1]).flatten()
    l_mean_s=mean_energy('z',shower_s).reshape([l,1]).flatten()
    data_l=[l_mean_g,l_mean_d,l_mean_i,l_mean_s]
    
    labels=['Geant4','CaloDiffusion','CaloINN','CaloScore']
    xy='a'
    plot_mean_energy(xy,data_a,dataset,labels)
    xy='r'
    plot_mean_energy(xy,data_r,dataset,labels)
    xy='z'
    plot_mean_energy(xy,data_l,dataset,labels)
    
    
    #return data_a,data_r, data_l,a,r,l
    
    
        

dataset='2'
prep_dataset(dataset)

dataset='3'
prep_dataset(dataset)





