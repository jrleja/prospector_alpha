#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 5760
### Partition or queue name
#SBATCH -p shared,conroy,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'simulate_prior'
#SBATCH --constraint="intel"
### output and error logs
#SBATCH -o simulate_prior_%a.out
#SBATCH -e simulate_prior_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
srun -n 1 --mpi=pmi2 python $APPS/prospector_alpha/code/simulate_sfh_prior.py \
--cluster_idx="${SLURM_ARRAY_TASK_ID}" \
--outfile="$APPS"/prospector_alpha/results/priors/