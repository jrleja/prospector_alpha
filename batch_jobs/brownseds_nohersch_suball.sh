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
#SBATCH -J 'nohersch'
### output and error logs
#SBATCH -o brownseds_nohersch_%a.out
#SBATCH -e brownseds_nohersch_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
### source activate pympi
mpirun -n 32 python $APPS/bsfh/demo/prospectr.py --param_file=$APPS/threedhst_bsfh/parameter_files/brownseds_nohersch/brownseds_nohersch_params_$SLURM_ARRAY_TASK_ID.py  --sps=fsps 