import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import torch
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse
from correlations_helper import *
import sys


parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                              'Fast Calorimeter Challenge 2022.'))

parser.add_argument('--input_file', '-i', help='Name of the input file to be evaluated.')
parser.add_argument('--reference_file', '-r',
                    help='Name and path of the .hdf5 file to be used as reference. ')

parser.add_argument('--mode', '-m', default='corr',choices=['corr','corr_dual','bob','bbb' ],
                    help=("what type of plots will be generated: "+\
                         "corr will generate a plot of layer to layer correlations for all layers"+\
                         "corr_dual will generate a side by side plot of partial layers to layers correlation for both generated and reference data"+\
                          "bob will call bar on bar which focuses one particular layer's correlation with all other layers"+\
                         "bbb will call bar by bar same as bob except the presentation is different."))

parser.add_argument('--dataset', '-d', choices=['1-photons', '1-pions', '2', '3'],
                    help='Which dataset is evaluated.')
parser.add_argument('--output_dir', default='evaluation_results/',
                    help='Where to store evaluation output files (plots and scores).')
parser.add_argument('--model_name', '-n', choices=['CaloDiffusion', 'CaloFlow', 'CaloScore', 'CaloINN'],
                    help='Which Gen AI model is evaluated.')

# CUDA parameters
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')


def l2l_photon(shower):
    layer1 = np.sum(shower[:,:8],-1,keepdims=True)

    layer2 = np.sum(shower[:,8:168],-1,keepdims=True)

    layer3 = np.sum(shower[:,168:358],-1,keepdims=True)

    layer4 = np.sum(shower[:,358:363],-1,keepdims=True)

    layer5 = np.sum(shower[:,363:],-1,keepdims=True)

    layer_photon = np.concatenate([layer1,layer2,layer3,layer4,layer5],-1)
    
    return layer_photon


def l2l_pion(shower):
    layer1 = np.sum(shower[:,:8],-1,keepdims=True)

    layer2 = np.sum(shower[:,8:108],-1,keepdims=True)

    layer3 = np.sum(shower[:,108:208],-1,keepdims=True)

    layer4 = np.sum(shower[:,208:213],-1,keepdims=True)

    layer5 = np.sum(shower[:,213:363],-1,keepdims=True)

    layer6 = np.sum(shower[:,363:523],-1,keepdims=True)

    layer7 = np.sum(shower[:,523:],-1,keepdims=True)

    layer_pion = np.concatenate([layer1,layer2,layer3,layer4,layer5,layer6,layer7],-1)
    
    return layer_pion

def process_layer_sum(shower,shape):
    samples=shower.shape[0]
    
    shower_ = shower.reshape(shape)
    layer_ = np.sum(shower_,(2,3,4),keepdims=True)
    layer_=layer_.reshape(samples,shape[1])
    
    return layer_


if __name__ == '__main__':

    args = parser.parse_args()

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        
    gen_Einc,gen_shower=file_read(args.input_file)
    
    ref_Einc,ref_shower=file_read(args.reference_file)
    
    if args.dataset=='2':
        
        shape=[-1,45,16,9,1]
        gen_layers=process_layer_sum(gen_shower,shape)
        ref_layers=process_layer_sum(ref_shower,shape)
        
    elif args.dataset=='3':
        
        shape=[-1,45,50,18,1]
        gen_layers=process_layer_sum(gen_shower,shape)
        ref_layers=process_layer_sum(ref_shower,shape)
        
    elif args.dataset=='1-photons':
        
        gen_layers=l2l_photon(gen_shower)
        ref_layers=l2l_photon(ref_shower)
        
    else:
        
        gen_layers=l2l_pion(gen_shower)
        ref_layers=l2l_pion(ref_shower)
        
        
    #calculate correlations
    
    gen_corr,gen_p=calculateCorrelation(gen_layers)
    ref_corr,ref_p=calculateCorrelation(ref_layers)
    
    if args.mode=='corr':
        file_name=f"l2l_corr_gen_DS_{args.dataset}_model_{args.model_name}.pdf"
        data_name=f"dataset_{args.dataset}"
        draw_heatmap(gen_corr,os.path.join(args.output_dir,file_name),data_name)
        
        file_name=f"l2l_corr_Geant4_DS_{args.dataset}.pdf"
        data_name=f"dataset_{args.dataset}"
        draw_heatmap(ref_corr,os.path.join(args.output_dir,file_name),data_name)
        
    elif args.mode=='corr_dual':
        for i in range(0,45,5):
            print(i,i+5)
            gen_corr_=gen_corr[i:i+5,:].T
            ref_corr_=ref_corr[i:i+5,:].T
            custom_xticks=list(range(i,i+5,1))
            fname=f"dual_heatmap_layer_{i}_to_layer_{i+4}.pdf"
            draw_heatmap_dual(gen_corr_,ref_corr_,os.path.join(args.output_dir,fname),\
                              f"{args.model_name}_DS_{args.dataset}",f"Geant4_DS_{args.dataset}",custom_xticks)
            
    elif args.mode=='bob':
        print("not implemented")
              
    else:
        print("not implemented")
        
        
        
        
    
