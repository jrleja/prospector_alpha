#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 5760
### Partition or queue name
#SBATCH -p conroy,shared,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'gama_logm'
### Only Intel chips
#SBATCH --constraint="intel"
### output and error logs
#SBATCH -o gama_logm_${3}.out
#SBATCH -e gama_logm_${3}.err
echo ${1} ${2}
srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/gama_logm_params.py \
--objname="${1}" \
--outfile="${2}"