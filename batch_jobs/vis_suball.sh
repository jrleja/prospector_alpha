#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p shared
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'vis'
### output and error logs
#SBATCH -o vis_%a.out
#SBATCH -e vis_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/vis_params.py \
--outfile="$APPS"/prospector_alpha/results/vis/vis_"${SLURM_ARRAY_TASK_ID}" \
--filter_key="${SLURM_ARRAY_TASK_ID}"