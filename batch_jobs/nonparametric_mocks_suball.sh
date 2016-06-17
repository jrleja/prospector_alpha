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
#SBATCH --mem-per-cpu=2500
### Job name
#SBATCH -J 'nonparametric_mocks'
### output and error logs
#SBATCH -o nonparametric_mocks_%a.out
#SBATCH -e nonparametric_mocks_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
### source activate pympi
mpirun -n 32 python $APPS/bsfh/demo/prospectr.py --param_file=$APPS/threedhst_bsfh/parameter_files/nonparametric_mocks/nonparametric_mocks_params_$SLURM_ARRAY_TASK_ID.py  --sps=fsps 