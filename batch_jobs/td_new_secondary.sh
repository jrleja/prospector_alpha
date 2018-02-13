#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 4
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p conroy,shared,serial_requeue
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'td_new_sec'
### output and error logs
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_new.ids"
x=`expr $SLURM_ARRAY_TASK_ID - 1`

for value in {1..4}
do
n=`expr $x \* 16 + $value`
OBJID=$(sed -n "${n}p" "$IDFILE")
FIELD=${OBJID%_*}

srun -n 1 --exclusive --mpi=pmi2 python $APPS/prospector_alpha/code/td/postprocessing.py \
$APPS/prospector_alpha/parameter_files/td_new_params.py \
--objname="$OBJID" \
--overwrite=True \
--plot=False \
--shorten_spec=True &
done
wait
