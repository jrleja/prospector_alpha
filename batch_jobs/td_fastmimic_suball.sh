#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p conroy,general,conroy-intel,serial_requeue
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'td_fastmimic'
### output and error logs
#SBATCH -o td_fastmimic_%a.out
#SBATCH -e td_fastmimic_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_fastmimic.ids"
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
srun -n $SLURM_NTASKS --mpi=pmi2 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file="$APPS"/prospector_alpha/parameter_files/td_fastmimic_params.py \
--objname="$OBJID" \
--outfile="$APPS"/prospector_alpha/results/td_fastmimic/"$OBJID"