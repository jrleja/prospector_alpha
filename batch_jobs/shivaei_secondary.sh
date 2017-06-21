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
#SBATCH -J 'shivaei_sec'
### output and error logs
#SBATCH -o shivaei_sec_%a.out
#SBATCH -e shivaei_sec_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/threedhst_bsfh/data/shivaei_ids.txt"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")

python $APPS/threedhst_bsfh/code/extra_output.py \
$APPS/threedhst_bsfh/parameter_files/shivaei_params.py \
--outname="$APPS"/threedhst_bsfh/results/shivaei/"$OBJID"