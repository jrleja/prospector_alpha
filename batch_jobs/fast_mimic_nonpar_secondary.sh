#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p conroy,shared,serial_requeue,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'fast_nonpar_sec'
### output and error logs
#SBATCH -o fast_nonpar_sec_%a.out
#SBATCH -e fast_nonpar_sec_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_new.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")

srun -n 1 --mpi=pmi2 python $APPS/prospector_alpha/code/td/postprocessing.py \
$APPS/prospector_alpha/parameter_files/fast_nonpar_params.py \
--objname="$OBJID" \
--overwrite=True \
--plot=False \
--shorten_spec=True
