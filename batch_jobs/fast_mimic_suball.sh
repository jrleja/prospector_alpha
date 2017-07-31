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
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_massive.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
srun -n $SLURM_NTASKS --mpi=pmi2 python $APPS/bsfh/scripts/prospector.py \
--param_file="$APPS"/prospector_alpha/parameter_files/fast_mimic_params.py \
--objname="$OBJID" \
--outfile="$APPS"/prospector_alpha/results/fast_mimic/"$OBJID"