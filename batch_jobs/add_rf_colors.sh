#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 1440
### Partition or queue name
#SBATCH -p itc_cluster,shared,conroy_requeue
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'rf_calc'
### output and error logs
#SBATCH -o rf_calc_%a.out
#SBATCH -e rf_calc_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
python $APPS/prospector_alpha/code/add_rf_colors.py \
--idx="${SLURM_ARRAY_TASK_ID}"
