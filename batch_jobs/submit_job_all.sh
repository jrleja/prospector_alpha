#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 32
### Requested number of nodes
#SBATCH -N 32
### Requested computing time in minutes
#SBATCH -t 60
### Partition or queue name
#SBATCH -p conroy
### memory per cpu, in MB
#SBATCH --mem-per-cpu=2000
### Job name
#SBATCH -J '3D_run$SLURM_ARRAY_TASK_ID'
### output and error logs
#SBATCH -o mpitest_$SLURM_ARRAY_TASK_ID.out
#SBATCH -e mpitest_$SLURM_ARRAY_TASK_ID.err
### source activate pympi
mpirun -n 32 python $APPS/bsfh/demo/prospectr.py --param_file=$APPS/threedhst_bsfh/parameter_files/threedhst_params_$SLURM_ARRAY_TASK_ID.py --sps=fsps --custom_filter_keys=$APPS/threedhst_bsfh/filters/filter_keys_threedhst.txt 