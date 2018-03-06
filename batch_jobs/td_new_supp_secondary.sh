#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p conroy-intel,shared,serial-requeue,conroy,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'td_new_supp_sec'
### output and error logs
#SBATCH -o td_new_supp_sec_%a.out
#SBATCH -e td_new_supp_sec_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_new_supp.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
python $APPS/prospector_alpha/code/td/postprocessing.py \
$APPS/prospector_alpha/parameter_files/td_new_params.py \
--objname="$OBJID" \
--overwrite=True \
--shorten_spec=True \
--runname="td_new_supp"
