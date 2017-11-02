#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 2
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p ozone
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'td'
### output and error logs
#SBATCH -o td_huge_%a.out
#SBATCH -e td_huge_%a.err
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_huge.ids"
let n1=SLURM_ARRAY_TASK_ID*2 
let n2=n1+1 
OBJID1=$(sed -n "${n1}p" "$IDFILE")
OBJID2=$(sed -n "${n2}p" "$IDFILE")

srun -n 1 --exclusive --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/td_huge_params.py \
--objname="$OBJID1" --dlogz_init=10000000000 \
--outfile="$APPS"/prospector_alpha/results/td_huge/"$OBJID1" &

srun -n 2 --exclusive --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/td_huge_params.py \
--objname="$OBJID2" --dlogz_init=10000000000 \
--outfile="$APPS"/prospector_alpha/results/td_huge/"$OBJID2" &
wait