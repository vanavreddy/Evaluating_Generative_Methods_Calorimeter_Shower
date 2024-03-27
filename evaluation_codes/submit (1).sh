#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=caloINN
#SBATCH -t 12:00:00
#SBATCH --mem=64000
#SBATCH -p bii-gpu
#SBATCH --gres=gpu
#SBATCH -A bii_nssac

module load anaconda/2023.07-py3.11 
conda activate caloflow

module load texlive/2023
export PATH=~/bin:$PATH
module load cuda/12.2.2
module load cudnn/8.9.4.25
module load apptainer

#python updated_evaluation.py
#python src/main.py params/pions.yaml -c

python evaluate.py -i '/scratch/fa7sa/IJCAI_experiment/Generated_shower/CaloDiffusion_10000_sample/test_ds2.h5' -r '/scratch/fa7sa/IJCAI_experiment/dataset_2/dataset_2_2.hdf5' -m 'hist-p' -d '2' --output_dir 'evaluation_results/' 