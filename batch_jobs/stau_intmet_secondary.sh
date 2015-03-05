#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 120
### Partition or queue name
#SBATCH -p conroy,general
### memory per cpu, in MB
#SBATCH --mem-per-cpu=2000
### Job name
#SBATCH -J 'stausec'
### output and error logs
#SBATCH -o stau_intmet_sec_%a.out
#SBATCH -e stau_intmet_sec_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
### source activate pympi
python $APPS/threedhst_bsfh/code/extra_output.py $APPS/threedhst_bsfh/parameter_files/stau_intmet/stau_intmet_params_$SLURM_ARRAY_TASK_ID.py 