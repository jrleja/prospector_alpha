#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p serial_requeue,conroy
### memory per cpu, in MB
#SBATCH --mem-per-cpu=6500
### Job name
#SBATCH -J 'fmimic_sec'
### output and error logs
#SBATCH -o fmimic_sec_%a.out
#SBATCH -e fmimic_sec_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
python $APPS/threedhst_bsfh/code/extra_output.py $APPS/threedhst_bsfh/parameter_files/fast_mimic/fast_mimic_params_$SLURM_ARRAY_TASK_ID.py --ir_priors=False --measure_spectral_features=False --mags_nodust=False