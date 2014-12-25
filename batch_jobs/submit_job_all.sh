#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 32
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 320
### Partition or queue name
#SBATCH -p conroy,general
### memory per cpu, in MB
#SBATCH --mem-per-cpu=1000
### Job name
#SBATCH -J 'bsfhgal'
### output and error logs
#SBATCH -o neboff_%a.out
#SBATCH -e neboff_%a.err
### mail
#SBATCH --mail-type=END 
#SBATCH --mail-user=joel.leja@gmail.com
### source activate pympi
mpirun -n 32 python $APPS/bsfh/demo/prospectr.py --param_file=$APPS/threedhst_bsfh/parameter_files/halpha_selected_params_$SLURM_ARRAY_TASK_ID.py --sps=fsps --custom_filter_keys=$APPS/threedhst_bsfh/filters/filter_keys_threedhst.txt