#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 5760
### Partition or queue name
#SBATCH -p shared, conroy_requeue, itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'uvj_dirichlet'
### output and error logs
#SBATCH -o uvj_dirichlet_%a.out
#SBATCH -e uvj_dirichlet_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/uvj_dirichlet_params.py \
--outfile="$APPS"/prospector_alpha/results/uvj_dirichlet/uvj_dirichlet_"${SLURM_ARRAY_TASK_ID}" \
--uvj_key="${SLURM_ARRAY_TASK_ID}"