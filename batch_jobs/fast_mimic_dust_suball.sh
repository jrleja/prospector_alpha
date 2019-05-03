#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 5760
### Partition or queue name
#SBATCH -p shared,itc_cluster,conroy_requeue
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### output and error logs
#SBATCH -o fast_mimic_dust_%a.out
#SBATCH -e fast_mimic_dust_%a.err
### Job name
#SBATCH -J 'fast_mimic_dust'
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_new.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/fast_mimic_dust_params.py \
--objname="$OBJID" \
--outfile="$APPS"/prospector_alpha/results/fast_mimic_dust/"$OBJID" \
--runname="td_new"
