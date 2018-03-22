#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p shared,conroy-intel,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=5000
### Job name
#SBATCH -J 'bseds_agn'
### output and error logs
#SBATCH -o bseds_agn_%a.out
#SBATCH -e bseds_agn_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/prospector_alpha/data/brownseds_agn.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
srun -n 1 --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/brownseds_agn_dynesty_params.py \
--objname="$OBJID" \
--outfile="$APPS"/prospector_alpha/results/brownseds_agn_dynesty/"$OBJID"
