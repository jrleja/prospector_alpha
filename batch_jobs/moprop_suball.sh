#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p conroy_requeue,shared,itc_cluster
### memopropry per cpu, in MB
#SBATCH --mem-per-cpu=4000
### output and error logs
#SBATCH -o moprop_%a.out
#SBATCH -e moprop_%a.err
### Job name
#SBATCH -J 'moprop'
srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/moprop_params.py \
--objname="$SLURM_ARRAY_TASK_ID" \
--outfile="$APPS"/prospector_alpha/results/moprop/"$SLURM_ARRAY_TASK_ID"