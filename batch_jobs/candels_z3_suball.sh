#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p conroy-intel,shared,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### output and error logs
#SBATCH -o candels_z3_%a.out
#SBATCH -e candels_z3_%a.err
### Job name
#SBATCH -J 'candels_z3'
IDFILE=$APPS"/prospector_alpha/data/CANDELS_GDSS_workshop_z3.dat"
n1=`expr $SLURM_ARRAY_TASK_ID + 1`
OBJID=$(awk -v i=$n1 -v j=1 'FNR == i {print $j}' $IDFILE)

srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/candels_z3_params.py \
--objname="$OBJID" \
--outfile="$APPS"/prospector_alpha/results/candels_z3/"$OBJID"