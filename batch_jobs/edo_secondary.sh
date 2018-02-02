#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 960
### Partition or queue name
#SBATCH -p conroy,serial_requeue,conroy-intel
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'edo_sec'
### output and error logs
#SBATCH -o edo_sec.out
#SBATCH -e edo_sec.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
python $APPS/prospector_alpha/code/td/postprocessing.py $APPS/prospector_alpha/parameter_files/edo/bns_params.py \
--shorten_spec=False --objname=bns_galaxy