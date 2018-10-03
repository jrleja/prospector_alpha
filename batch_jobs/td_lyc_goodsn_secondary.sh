#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p conroy_requeue,serial_requeue,itc_cluster,shared
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'td_lyc_sec'
### output and error logs
#SBATCH -o td_lyc_sec_%a.out
#SBATCH -e td_lyc_sec_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDX=`expr $SLURM_ARRAY_TASK_ID + 1`
IDFILE="$APPS"/prospector_alpha/data/3dhst/joel_goodsn.csv
OBJNAME=GOODSN_$(awk -F, -v i=$IDX -v j=3 'FNR == i {print $j}' $IDFILE)
srun -n 1 --exclusive --mpi=pmi2 python $APPS/prospector_alpha/code/td/postprocessing.py \
$APPS/prospector_alpha/parameter_files/td_lyc_params.py \
--objname="${OBJNAME}" \
--overwrite=True \
--plot=True \
--shorten_spec=True