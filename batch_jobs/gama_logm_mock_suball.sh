#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 7200
### Partition or queue name
#SBATCH -p conroy-intel,shared,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### output and error logs
#SBATCH -o gama_logm_mock_%a.out
#SBATCH -e gama_logm_mock_%a.err
### Job name
#SBATCH -J 'gama_logm_mock'
x=`expr $SLURM_ARRAY_TASK_ID + 4`
srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/gama_logm_mock_params.py \
--mock_key="${x}" \
--outfile="$APPS"/prospector_alpha/results/gama_logm_mock/"${SLURM_ARRAY_TASK_ID}"