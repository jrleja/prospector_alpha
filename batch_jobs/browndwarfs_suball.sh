#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 32
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 1440
### Partition or queue name
#SBATCH -p conroy,general
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'browndwarfs'
### output and error logs
#SBATCH -o browndwarfs_%a.out
#SBATCH -e browndwarfs_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
srun -n $SLURM_NTASKS --mpi=pmi2 python $APPS/prospector/scripts/prospector.py --param_file=$APPS/prospector_alpha/parameter_files/browndwarfs/browndwarfs_params_$SLURM_ARRAY_TASK_ID.py