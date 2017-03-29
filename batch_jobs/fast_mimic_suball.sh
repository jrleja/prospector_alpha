#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 32
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 1440
### Partition or queue name
#SBATCH -p conroy-intel,conroy,general
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'fmimic'
### output and error logs
#SBATCH -o fmimic_%a.out
#SBATCH -e fmimic_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
srun -n $SLURM_NTASKS --mpi=pmi2 python $APPS/bsfh/scripts/prospector.py --param_file=$APPS/threedhst_bsfh/parameter_files/fast_mimic/fast_mimic_params_$SLURM_ARRAY_TASK_ID.py 
