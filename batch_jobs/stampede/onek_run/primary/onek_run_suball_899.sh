#!/bin/bash
### Name of the job 
### Requested number of nodes
#SBATCH -n 32
### Requested computing time in minutes
#SBATCH -t 1:00:00
###partition
#SBATCH -p normal
### Job name
#SBATCH -J '3Drun'
### output and error logs
#SBATCH -o onek_run_899.out
#SBATCH -e onek_run_899.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
ibrun -np 32 python $APPS/bsfh/demo/prospectr.py --param_file=$APPS/threedhst_bsfh/parameter_files/onek_run/onek_run_params_899.py  --sps=fsps --custom_filter_keys=$APPS/threedhst_bsfh/filters/filter_keys_threedhst.txt 
