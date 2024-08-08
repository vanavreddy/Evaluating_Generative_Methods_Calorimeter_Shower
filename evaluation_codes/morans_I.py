import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import pysal.lib
from splot.esda import moran_scatterplot
import matplotlib.pyplot as plt
from esda.moran import Moran

def file_read(file_name):
    with h5py.File(file_name, "r") as h5f:
        e = h5f['incident_energies'][::].astype(np.float32)  
        shower = h5f['showers'][::].astype(np.float32)
        
    return e, shower


g_path='/project/biocomplexity/fa7sa/calorimeter_evaluation_data/dataset_2/dataset_2_electron_Geant4.h5'
d_path='/project/biocomplexity/fa7sa/calorimeter_evaluation_data/dataset_2/dataset_2_electron_CaloDiffusion.h5'
i_path='/project/biocomplexity/fa7sa/calorimeter_evaluation_data/dataset_2/dataset_2_electron_CaloINN.h5'
s_path='/project/biocomplexity/fa7sa/calorimeter_evaluation_data/dataset_2/dataset_2_electron_CaloScore.h5'
e_g,shower_g=file_read(g_path)
e_d,shower_d=file_read(d_path)
e_i,shower_i=file_read(i_path)
e_s,shower_s=file_read(s_path)

shape=[-1,45,16,9]
shower_s = shower_s.reshape(shape)
shower_g=shower_g.reshape(shape)
shower_d=shower_d.reshape(shape)
shower_i=shower_i.reshape(shape)
def create_custom_weight_matrix(nrows, ncols):
    weights = {}
    for row in range(nrows):
        for col in range(ncols):
            index = row * ncols + col
            neighbors = []
            
            # Previous row
            prev_row = (row - 1) % nrows
            neighbors.append(prev_row * ncols + col)
            
            # Next row
            next_row = (row + 1) % nrows
            neighbors.append(next_row * ncols + col)
            
            # Previous column
            if col > 0:
                neighbors.append(row * ncols + (col - 1))
                
            # Next column
            if col < ncols - 1:
                neighbors.append(row * ncols + (col + 1))
                
            # Construct the dictionary for the current cell
            weights[index] = {neighbor: 1 for neighbor in neighbors}
    
    return weights

def calculate_morans_i(array, weight_matrix,model=None,layer=22):
    
    y = array.flatten()
    
    # Ensure the custom weight matrix is in the correct format
    weight_matrix = pysal.lib.weights.weights.W(weight_matrix)
    
    # Calculate Moran's I
    morans_i = Moran(y, weight_matrix)
    return morans_i


def draw_scatterplot(morans_i,model,layer):
    
    fig, ax = moran_scatterplot(morans_i, aspect_equal=True)

    print("Raster Dimensions:\t" + str(array.shape))
    print("Moran's I Value:\t" +str(round(morans_i.I,4)))
    plt.savefig(f'morans_I_{model}_{layer}.png')
    plt.show()

   
    
nrows=16
ncols=9
custom_weights = create_custom_weight_matrix(nrows, ncols)


### draw scatterplot for particular sample and particular layer number
array_g=shower_g[50000,22,:,:]
#print(array_g.shape)
array_d=shower_d[50000,22,:,:]
array_i=shower_i[50000,22,:,:]
array_s=shower_s[50000,22,:,:]
ms=calculate_morans_i(array_s,custom_weights,'caloscore_50000_',layer=22)
mg=calculate_morans_i(array_g, custom_weights,'Geant4_50000_',layer=22)
md=calculate_morans_i(array_d, custom_weights,'Calodiff_50000_',layer=22)
mi=calculate_morans_i(array_i, custom_weights,'CaloINN_50000_',layer=22)

draw_scatterplot(ms,'CaloScore','22')
draw_scatterplot(mg,'Geant4','22')
draw_scatterplot(mi,'CaloINN','22')
draw_scatterplot(md,'CaloDiff','22')

#array=np.sum(shower_s,axis=1)
array_s=shower_s[:,22,:,:] 
MI=[]
for sample in range(array_s.shape[0]):
    
    array=array_s[sample,:,:]
    #print(array.shape)
    morans_i=calculate_morans_i(array,custom_weights)
    MI.append(morans_i.I)
    
#Cause NAN.... Use MI for Histogram plot  
# avg_MI=np.mean(np.array(MI))

# print(round(avg_MI,4))
    