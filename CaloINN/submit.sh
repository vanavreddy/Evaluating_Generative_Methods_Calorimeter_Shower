#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=caloINN
#SBATCH -t 50:00:00
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

python src/main.py params/example_ds3.yaml -c
#python src/main.py params/pions.yaml -c
