#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p conroy-intel,shared,serial-requeue,conroy
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'uvj_sec'
### output and error logs
#SBATCH -o uvj_sec_%a.out
#SBATCH -e uvj_sec_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
python $APPS/prospector_alpha/code/td/postprocessing.py \
$APPS/prospector_alpha/parameter_files/uvj_params.py \
--objname=uvj_"${SLURM_ARRAY_TASK_ID}" \
--overwrite=True \
--shorten_spec=True \
--runname="uvj"
