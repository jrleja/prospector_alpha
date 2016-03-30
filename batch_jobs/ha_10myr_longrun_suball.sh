#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 32
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 12000
### Partition or queue name
#SBATCH -p conroy,general
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'ha_10myr'
### output and error logs
#SBATCH -o ha_10myr_longrun_%a.out
#SBATCH -e ha_10myr_longrun_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
### source activate pympi
mpirun -n 32 python $APPS/bsfh/demo/prospectr.py --param_file=$APPS/threedhst_bsfh/parameter_files/ha_10myr_longrun/ha_10myr_longrun_params_$SLURM_ARRAY_TASK_ID.py  --sps=fsps 