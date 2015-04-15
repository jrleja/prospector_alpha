#!/bin/bash
### Name of the job 
### Requested number of nodes
#SBATCH -n 32
### Requested computing time in minutes
#SBATCH -t 10:00:00
###partition
#SBATCH -p normal
### memory per cpu, in MB
#SBATCH --mem-per-cpu=1200
### Account
### PHAT
#SBATCH -A TG-AST130057
### Job name
#SBATCH -J '3Drun'
### output and error logs
#SBATCH -o dtau_genpop_zperr_%a.out
#SBATCH -e dtau_genpop_zperr_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
ibrun python-mpi $APPS/bsfh/demo/prospectr.py --param_file=$APPS/threedhst_bsfh/parameter_files/dtau_genpop_zperr/dtau_genpop_zperr_params_1.py  --sps=fsps --custom_filter_keys=$APPS/threedhst_bsfh/filters/filter_keys_threedhst.txt 