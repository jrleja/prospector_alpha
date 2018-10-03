#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 5760
### Partition or queue name
#SBATCH -p conroy_requeue,shared,itc_cluster
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'td_lyc'
### output and error logs
#SBATCH -o td_lyc_goodss_%a.out
#SBATCH -e td_lyc_goodss_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDX=`expr $SLURM_ARRAY_TASK_ID + 1`
IDFILE="$APPS"/prospector_alpha/data/3dhst/joel_goodss.csv
OBJNAME=GOODSS_$(awk -F, -v i=$IDX -v j=3 'FNR == i {print $j}' $IDFILE)
srun -n $SLURM_NTASKS --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/td_lyc_params.py \
--objname="${SLURM_ARRAY_TASK_ID}" \
--outfile="$APPS"/prospector_alpha/results/td_lyc/"${OBJNAME}"