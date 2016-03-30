#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 1000
### Partition or queue name
#SBATCH -p conroy,serial_requeue
### memory per cpu, in MB
#SBATCH --mem-per-cpu=6000
### Job name
#SBATCH -J 'brownsedsec'
### output and error logs
#SBATCH -o brownseds_longrun_sec_%a.out
#SBATCH -e brownseds_longrun_sec_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
### source activate pympi
python $APPS/threedhst_bsfh/code/extra_output.py $APPS/threedhst_bsfh/parameter_files/brownseds_longrun/brownseds_longrun_params_$SLURM_ARRAY_TASK_ID.py 