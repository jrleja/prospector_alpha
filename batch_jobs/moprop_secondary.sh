#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p shared,serial_requeue,itc_cluster,conroy_requeue
### memopropry per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'moprop_sec'
### output and error logs
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
srun -n 1 --exclusive --mpi=pmi2 python $APPS/prospector_alpha/code/td/postprocessing.py \
$APPS/prospector_alpha/parameter_files/moprop_params.py \
--objname="$SLURM_ARRAY_TASK_ID" \
--overwrite=True \
--plot=True \
--shorten_spec=True
