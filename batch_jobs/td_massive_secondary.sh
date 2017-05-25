#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p conroy,serial_requeue
### memory per cpu, in MB
#SBATCH --mem-per-cpu=6500
### Job name
#SBATCH -J '3d_mass_sec'
### output and error logs
#SBATCH -o td_massive_sec_%a.out
#SBATCH -e td_massive_sec_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/threedhst_bsfh/data/3dhst/td_massive.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")

python $APPS/threedhst_bsfh/code/extra_output.py 
$APPS/threedhst_bsfh/parameter_files/td_massive/td_massive_params.py \
--outname="$APPS"/threedhst_bsfh/results/td_massive/"$OBJID"