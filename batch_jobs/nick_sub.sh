#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 32
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 1440
### Partition or queue name
#SBATCH -p conroy,general,conroy-intel
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'nick'
### output and error logs
#SBATCH -o nick.out
#SBATCH -e nick.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
srun -n $SLURM_NTASKS --mpi=pmi2 python $APPS/prospector/scripts/prospector.py --param_file=$APPS/prospector_alpha/parameter_files/nick/nick_params.py 
