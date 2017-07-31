#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 2880
### Partition or queue name
#SBATCH -p conroy-intel,conroy,general,serial_requeue
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'brownnest'
### output and error logs
#SBATCH -o brownseds_agn_nestle_%a.out
#SBATCH -e brownseds_agn_nestle_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/prospector_alpha/data/brownseds_agn.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
python $APPS/prospector/scripts/prospector_nest.py \
--param_file="$APPS"/prospector_alpha/parameter_files/brownseds_agn_nestle/brownseds_agn_nestle_params.py \
--objname="$OBJID" \
--outfile="$APPS"/prospector_alpha/results/brownseds_agn_nestle/brownseds_agn_nestle_"$OBJID"