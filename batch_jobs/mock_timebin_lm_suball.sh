#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 7200
### Partition or queue name
#SBATCH -p conroy,shared
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### output and error logs
#SBATCH -o mock_timebin_lm_%a.out
#SBATCH -e mock_timebin_lm_%a.err
### Job name
#SBATCH -J 'mock_timebin_lm'
### chip type
#SBATCH --constraint="intel"
srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/mock_timebin_lm_params.py \
--objname="${SLURM_ARRAY_TASK_ID}" \
--outfile="$APPS"/prospector_alpha/results/mock_timebin_lm/"${SLURM_ARRAY_TASK_ID}"