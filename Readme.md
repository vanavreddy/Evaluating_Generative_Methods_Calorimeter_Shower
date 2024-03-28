This project is inspired by CaloChallenge 2022[https://calochallenge.github.io/homepage/]. 

To generate the correlation graphs:
1. module load anaconda
   
2. conda activate caloflow
   
3. This is only applicable for Dataset 2 and Dataset 3.

python correlate.py -i /scratch/fa7sa/IJCAI_experiment/Generated_shower/CaloDiffusion_10000_sample/test_ds2.h5 -r /scratch/fa7sa/IJCAI_experiment/dataset_2/dataset_2_2.hdf5 -d '2' -m 'corr_dual' -n 'CaloDiffusion'

4. Applicable for all dataset.

 python correlate.py -i /scratch/fa7sa/IJCAI_experiment/Generated_shower/CaloDiffusion_10000_sample/test_ds2.h5 -r /scratch/fa7sa/IJCAI_experiment/dataset_2/dataset_2_2.hdf5 -d '2' -m 'corr' -n 'CaloDiffusion'


