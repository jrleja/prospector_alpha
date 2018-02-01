#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 5760
### Partition or queue name
#SBATCH -p shared,conroy-intel
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'bns'
### output and error logs
#SBATCH -o edo_%a.out
#SBATCH -e edo_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
python $APPS/prospector/scripts/prospector_dynesty.py --param_file=$APPS/prospector_alpha/parameter_files/bns_params.py 