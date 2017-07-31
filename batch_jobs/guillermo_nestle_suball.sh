#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 2880
### Partition or queue name
#SBATCH -p conroy-intel
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'guilnest'
### output and error logs
#SBATCH -o guillermo_nestle.out
#SBATCH -e guillermo_nestle.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/prospector_alpha/data/brownseds_agn.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
python $APPS/prospector/scripts/prospector_nest.py \
--param_file="$APPS"/prospector_alpha/parameter_files/guillermo_nestle/guillermo_nestle_params.py