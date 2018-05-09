#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p shared,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### output and error logs
#SBATCH -o td_shivaei_%a.out
#SBATCH -e td_shivaei_%a.err
### Job name
#SBATCH -J 'td_shivaei'
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_shivaei.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/td_shivaei_params.py \
--objname="$OBJID" \
--outfile="$APPS"/prospector_alpha/results/td_shivaei/"$OBJID"