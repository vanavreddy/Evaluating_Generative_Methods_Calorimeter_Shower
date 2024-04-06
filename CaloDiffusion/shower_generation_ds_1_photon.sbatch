#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=calodiff
#SBATCH -t 10:00:00
#SBATCH --mem=64000
#SBATCH -p bii-gpu
#SBATCH --gres=gpu
#SBATCH -A bii_nssac
module load anaconda/2023.07-py3.11
source activate /project/biocomplexity/fa7sa/CaloDiffusionPaper/env_calodiffusion
python plot.py --config /project/biocomplexity/fa7sa/CaloDiffusionPaper/configs/config_dataset1_photon.json --model_loc /project/biocomplexity/fa7sa/CaloDiffusionPaper/trained_models/dataset1_photon.pth --sample --sample_steps 400 --sample_offset 2 --nevts 121000 -g test_ds1_photon_121000.h5 --plot_folder /project/biocomplexity/fa7sa/CaloDiffusionPaper/shower_gen --data_folder /project/biocomplexity/fa7sa/IJCAI_experiment/dataset_1