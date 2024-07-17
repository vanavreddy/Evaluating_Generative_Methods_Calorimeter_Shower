import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

def file_read(file_name):
    with h5py.File(file_name, "r") as h5f:
        e = h5f['incident_energies'][::].astype(np.float32)  
        shower = h5f['showers'][::].astype(np.float32)
        
    return e, shower

def cc_radial_layer(sample):
    
    all_rs=[]
    for r in range(9):

        data=sample[:,:,:,r]
        #sh=[-1,45,16]
        data=data.reshape([-1,45,16])
        final_corr=[]
        final_fisher=[]
        for i in range(16):
            corr_layer=np.zeros((45,45))
            fisher_z_corr=np.zeros((45,45))
            for j in range(45):
                for k in range(45):
                    corr_layer[j,k],_=pearsonr(data[:,j,i],data[:,k,i])
                    fisher_z_corr[j,k] = fisher_z(corr_layer[j,k])


            final_fisher.append(fisher_z_corr)
            final_corr.append(corr_layer)


        final_corr=np.array(final_corr)
        final_fisher=np.array(final_fisher)

        avg_fisher=np.mean(final_fisher, axis=0)
        avg_corr_g=inverse_fisher_z(avg_fisher)
        all_rs.append(avg_corr_g)
        
    return np.array(all_rs)


def draw_heatmap_angular_var(average_correlation_matrix, model_name, dataset,label):
    rows = 3
    cols = 3

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))  # Adjust figsize as needed
    vmin = np.min(average_correlation_matrix)
    vmax = np.max(average_correlation_matrix)
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through each heatmap and plot it
    for i in range(9):
        ax = axes[i]
        heatmap = ax.imshow(average_correlation_matrix[i], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"Radial Bin {i + 1}")
        ax.set_xlabel(f'{label}')
        ax.set_ylabel(f'{label}')
        ax.invert_yaxis()
        fig.colorbar(heatmap, ax=ax)

    # Adjust layout

    plt.tight_layout()
    #plt.gca().invert_yaxis()
    plt.show()

    # Save the figure if needed
    fig.savefig(f'avg_corr_heatmaps_angular_variance_{model_name}_for_dataset_{dataset}_{label}.png', bbox_inches='tight', dpi=300)
    
def draw_heatmap_radial_var(average_correlation_matrix, model_name, dataset):    
    
    rows = 4
    cols = 4

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))  # Adjust figsize as needed
    vmin = np.min(average_correlation_matrix)
    vmax = np.max(average_correlation_matrix)
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through each heatmap and plot it
    for i in range(16):
        ax = axes[i]
        heatmap = ax.imshow(average_correlation_matrix[i], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"Angular Bin {i + 1}")
        ax.set_xlabel('Radial Bin')
        ax.set_ylabel('Radial Bin')
        ax.invert_yaxis()
        # ax.grid(True)


        # ax.set_xticks(np.arange(average_correlation_matrix_g.shape[2]))
        # ax.set_yticks(np.arange(average_correlation_matrix_g.shape[1]))
        # ax.set_xticklabels(np.arange(average_correlation_matrix_g.shape[2]))
        # ax.set_yticklabels(np.arange(average_correlation_matrix_g.shape[1]))
        # ax.xaxis.set_tick_params(rotation=0)  # Adjust tick rotation if needed
        fig.colorbar(heatmap, ax=ax)

    # Adjust layout

    plt.tight_layout()
    #plt.gca().invert_yaxis()
    plt.show()

    # Save the figure if needed
    fig.savefig(f'avg_corr_heatmaps_Radial_variance_{model_name}_for_dataset_{dataset}.png', bbox_inches='tight', dpi=300)


def cc_angular_calc_update(sample):
    cc_angular_layer = []
    p_angular_layer = []
    fisher_z_layer = []

    for layer in range(45):
        print(f"Layer {layer}")
        cc_angular = [] 
        p_angular = []
        fisher_z_angular = []
        
        for i in range(9):
            data = sample[:, layer, :, i]
            dim = data.shape[1]
            correlation_matrix = np.ones((dim, dim))
            p_value_matrix = np.zeros((dim, dim))
            fisher_z_matrix = np.zeros((dim, dim))

            for j in range(dim):
                for k in range(dim):
                    corr, p_value = pearsonr(data[:, j], data[:, k])
                    correlation_matrix[j, k] = corr
                    p_value_matrix[j, k] = p_value
                    fisher_z_matrix[j, k] = fisher_z(corr)
            
            cc_angular.append(correlation_matrix)
            p_angular.append(p_value_matrix)
            fisher_z_angular.append(fisher_z_matrix)
        
        cc_angular_layer.append(cc_angular)
        p_angular_layer.append(p_angular)
        fisher_z_layer.append(fisher_z_angular)
    
    # Convert lists to numpy arrays
    cc_angular_layer = np.array(cc_angular_layer)
    p_angular_layer = np.array(p_angular_layer)
    fisher_z_layer = np.array(fisher_z_layer)

    # Compute the average Fisher Z-transformed correlation coefficient across all layers
    average_fisher_z_matrix = np.mean(fisher_z_layer, axis=0)

    # Apply the inverse Fisher Z transformation to get the average correlation coefficient
    average_correlation_matrix = inverse_fisher_z(average_fisher_z_matrix)
    
    return cc_angular_layer, p_angular_layer, average_correlation_matrix

def cc_radial_calc_update(sample):
    cc_radial_layer=[]
    p_radial_layer=[]
    fisher_z_layer=[]
    for layer in range(45):
        print(f"Layer {layer}")
        cc_radial=[] 
        p_radial=[]
        fisher_z_radial=[]
        for i in range(16):
            data=sample[:,layer,i,:]
            #print(data.shape)
            dim=data.shape[1]
            correlation_matrix = np.ones((dim, dim))
            p_value_matrix = np.zeros((dim, dim))
            fisher_z_matrix = np.zeros((dim, dim))
            for j in range(dim):
                for k in range(dim):

                    corr, p_value = pearsonr(data[:, j], data[:, k])
                    correlation_matrix[j, k] = corr
                    p_value_matrix[j, k] = p_value
                    fisher_z_matrix[j, k] = fisher_z(corr)

            
            cc_radial.append(correlation_matrix)
            p_radial.append(p_value_matrix)
            fisher_z_radial.append(fisher_z_matrix)
        cc_radial_layer.append(cc_radial)
        p_radial_layer.append(p_radial)
        fisher_z_layer.append(fisher_z_radial)
        
    fisher_z_layer = np.array(fisher_z_layer)

    # Compute the average Fisher Z-transformed correlation coefficient across all layers
    average_fisher_z_matrix = np.mean(fisher_z_layer, axis=0)

    # Apply the inverse Fisher Z transformation to get the average correlation coefficient
    average_correlation_matrix = inverse_fisher_z(average_fisher_z_matrix)
        
    return np.array(cc_radial_layer),np.array(p_radial_layer),average_correlation_matrix

def fisher_z(r):
    return np.arctanh(r)  # This is equivalent to (1/2) * ln((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return np.tanh(z)  # This is equivalent to (e^(2z) - 1) / (e^(2z) + 1)

def all_neighboring_voxels(shower,model_name):
    print(shower.shape)
    layer_voxels=[]
    for l in range(45):
        #print(f"current layer{l}")
        all_voxel=np.zeros((16,9))
        for r in range(9):
            #print(f"r is {r}")
            for a in range(16):
                #print(f"a is {a}")
                idx=find_neighbors(a,r)
                # if pearson correlation is using
                fisher_z_inner=neighboring_correlation(shower, l, idx,a,r)
                #print(fisher_z_inner.shape)
                fisher_avg=np.mean(fisher_z_inner)
                fisher_inv=inverse_fisher_z(fisher_avg)
                all_voxel[a,r]=fisher_inv
                
                # #if concordance correlation coefficient is used:
                # corr_matrix=concordaneCC(shower,l,a,r,idx)
                # all_voxel[a,r]=np.mean(corr_matrix)


        layer_voxels.append(all_voxel)

    layer_voxels=np.array(layer_voxels)
    print(layer_voxels.shape)
    fn=f"PCC_all_neighbors_{model_name}.npy"
    np.save(fn, layer_voxels)
    
    #model_name=model_name
    draw_heatmaps_perLayerAllVoxels(layer_voxels,model_name)
    return layer_voxels
def neighboring_correlation(shower, l, idx,a,r):
    
    if l!=0 and l!=44:
        layers=[l-1,l,l+1]
    else:
        if l==0:
            layers=[l,l+1]
        elif l==44:
            layers=[l-1,l]
    corr_matrix=[]
    fisher_z_inner=[]
    for i in idx:
        i_a=i[0]
        i_r=i[1]
        #print(a,r,i_a,i_r)
        for i_l in layers:
            if i_l== l and i_a==a and i_r==r:
                continue 
            crr,_=pearsonr(shower[:,l,a,r],shower[:,i_l,i_a,i_r])
            corr_matrix.append(crr)
            fisher_z_inner.append(fisher_z(crr))
    #print(fisher_z_inner.shape)
    #print(f"in neighboring function {len(fisher_z_inner)}")
    return np.array(fisher_z_inner)
def draw_heatmaps_perLayerAllVoxels(layer_voxels,model_name):
    # Create a figure with 3 rows and 3 columns of subplots
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    global_min = np.min(layer_voxels)
    global_max = np.max(layer_voxels)
    # Iterate over each subplot
    for i in range(9):
        # Determine the row and column of the subplot
        row = i // 3
        col = i % 3

        # Extract the data for the current subplot
        subplot_data = layer_voxels[:, :, i]

        # Create the heatmap for the current subplot
        im = axs[row, col].imshow(subplot_data.T, aspect='auto', cmap='coolwarm',vmin=global_min, vmax=global_max)

        # Add a colorbar to the subplot
        fig.colorbar(im, ax=axs[row, col])

        # Set labels and title
        axs[row, col].set_xlabel('Layers')
        axs[row, col].set_ylabel('Angular Bins')
        axs[row, col].set_title(f'Radial Bin {i+1}')

        # Show grid
        axs[row, col].grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)

        # Set x-ticks and y-ticks
        axs[row, col].set_xticks(np.arange(0, 45, step=5))
        axs[row, col].set_yticks(np.arange(0, 16, step=2))

        # Invert y-axis to have the first row at the top
        axs[row, col].invert_yaxis()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.savefig(f"per_layer_all_neighboring_voxels_{model_name}.png",bbox_inches='tight', dpi=300)

def find_neighbors(a,r):
    if r!=0 and r!=8:
            idx_r=[r+1,r,r-1]
    else:
        if r==8:
            idx_r=[r,r-1]
        else: idx_r=[r+1,r]
            
    if a!=0 and a!=15:
        idx_a=[a+1,a,a-1]
    else:
        if a==0:
            idx_a=[a+1,a,15]
        if a==15:
            idx_a=[0,a,a-1]
                    
    index_pairs = [[i, j] for i in idx_a for j in idx_r]
    #index_pairs = [[i, j] for i, j in index_pairs if [i, j] != [a, r]]
    return index_pairs
        
def ang_radial_PCC(shower_g,shower_d, shower_i, shower_s):
    
    cc_r_g,p_r_g, avg_r_g=cc_radial_calc_update(shower_g)
    cc_r_d,p_r_d, avg_r_d=cc_radial_calc_update(shower_d)
    cc_r_i,p_r_i, avg_r_i=cc_radial_calc_update(shower_i)
    cc_r_s,p_r_s, avg_r_s=cc_radial_calc_update(shower_s)
    
    cc_a_g,p_a_g, avg_a_g=cc_angular_calc_update(shower_g)
    cc_a_d,p_a_d, avg_a_d=cc_angular_calc_update(shower_d)
    cc_a_i,p_a_i, avg_a_i=cc_angular_calc_update(shower_i)
    cc_a_s,p_a_s, avg_a_s=cc_angular_calc_update(shower_s)
    
    draw_heatmap_angular_var(avg_a_g,'Geant4', dataset='2')
    draw_heatmap_angular_var(avg_a_d,'CaloDiff', dataset='2')
    draw_heatmap_angular_var(avg_a_i,'CaloINN', dataset='2')
    draw_heatmap_angular_var(avg_a_s,'CaloScore', dataset='2')
    
    
    draw_heatmap_radial_var(avg_r_g,'Geant4', dataset='2')
    draw_heatmap_radial_var(avg_r_d,'CaloDiff', dataset='2')
    draw_heatmap_radial_var(avg_r_i,'CaloINN', dataset='2')
    draw_heatmap_radial_var(avg_r_s,'CaloScore', dataset='2')
    
def main():
    g_path='/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/dataset_2_electron_Geant4.h5'
    d_path='/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/dataset_2_electron_CaloDiffusion.h5'
    i_path='/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/dataset_2_electron_CaloINN.h5'
    s_path='/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/dataset_2_electron_CaloScore.h5'
    
    e_g,shower_g=file_read(g_path)
    e_d,shower_d=file_read(d_path)
    e_i,shower_i=file_read(i_path)
    e_s,shower_s=file_read(s_path)
    shape=[-1,45,16,9]
    #shape=[-1,45,144]
    samples=shower_g.shape[0]

    shower_g = shower_g.reshape(shape)
    shower_d=shower_d.reshape(shape)
    shower_i=shower_i.reshape(shape)
    shower_s=shower_s.reshape(shape)
    
    #ang_radial_PCC(shower_g, shower_d, shower_i, shower_s)
    
    layer_voxel_g=all_neighboring_voxels(shower_g, 'Geant4')
    layer_voxel_d=all_neighboring_voxels(shower_d,'CaloDiffusion')
    layer_voxel_i=all_neighboring_voxels(shower_i,'CaloINN')
    layer_voxel_s=all_neighboring_voxels(shower_s,'CaloScore')
    
    
    
    # rl_g=cc_radial_layer(shower_g)
    # rl_d=cc_radial_layer(shower_d)
    # rl_i=cc_radial_layer(shower_i)
    # rl_s=cc_radial_layer(shower_s)
    # draw_heatmap_angular_var(rl_g, 'Geant4',2,"Layers")
    # draw_heatmap_angular_var(rl_d, 'CaloDiffusion',2,"Layers")
    # draw_heatmap_angular_var(rl_i, 'CaloINN',2,"Layers")
    # draw_heatmap_angular_var(rl_s, 'CaloScore',2,"Layers")
    
    
    
    
    
    
main()