#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 5760
### Partition or queue name
#SBATCH -p conroy,shared,conroy-intel,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'gama_logm'
### output and error logs
#SBATCH -o gama_logm_%a.out
#SBATCH -e gama_logm_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/prospector_alpha/data/gama.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
srun -n $SLURM_NTASKS --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/gama_logm_params.py \
--objname="$OBJID" \
--outfile="$APPS"/prospector_alpha/results/gama_logm/"$OBJID"