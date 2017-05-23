#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 8
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 800
### Partition or queue name
#SBATCH -p conroy-intel,conroy,general
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'fmimic'
### output and error logs
#SBATCH -o fmimic_%a.out
#SBATCH -e fmimic_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/threedhst_bsfh/data/3dhst/COSMOS_td_massive.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
python $APPS/bsfh/scripts/prospector.py \
--param_file="$APPS"/threedhst_bsfh/parameter_files/fast_mimic_params.py \
--objname="$OBJID" \
--outfile="$APPS"/threedhst_bsfh/results/fast_mimic/"$OBJID"