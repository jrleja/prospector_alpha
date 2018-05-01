#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p conroy,shared,serial_requeue,itc_cluster,conroy-intel
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'brownseds_highz_freeq_sec'
### output and error logs
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/prospector_alpha/data/brownseds_agn.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")

srun -n 1 --mpi=pmi2 python $APPS/prospector_alpha/code/td/postprocessing.py \
$APPS/prospector_alpha/parameter_files/brownseds_highz_freeq_params.py \
--objname="$OBJID" \
--overwrite=True \
--plot=False \
--shorten_spec=True
