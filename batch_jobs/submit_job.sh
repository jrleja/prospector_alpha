#!/bin/bash
### Name of the job
### Requested number of nodes
#SBATCH -n 32
### Requested computing time in minutes
#SBATCH -t 60
### Partition or queue name
#SBATCH -p conroy
### memory per cpu, in MB
#SBATCH --mem-per-cpu=2000
### Job name
#SBATCH -J 'threedhst_run1'
### output and error logs
#SBATCH -o mpitest_%j.out
#SBATCH -e mpitest_%j.err
### source activate pympi
mpirun -n 32 python $APPS/bsfh/demo/prospectr.py --param_file=$APPS/threedhst_bsfh/parameter_files/threedhst_params.py --sps=fsps --custom_filter_keys=$APPS/threedhst_bsfh/filters/filter_keys_threedhst.txt 