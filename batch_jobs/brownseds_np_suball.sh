#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 32
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 720
### Partition or queue name
#SBATCH -p conroy,general
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'bseds_np'
### output and error logs
#SBATCH -o brownseds_np_%a.out
#SBATCH -e brownseds_np_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
### source activate pympi
mpirun -n $SLURM_NPROCS python $APPS/bsfh/scripts/prospector.py --param_file=$APPS/threedhst_bsfh/parameter_files/brownseds_np/brownseds_np_params_$SLURM_ARRAY_TASK_ID.py 