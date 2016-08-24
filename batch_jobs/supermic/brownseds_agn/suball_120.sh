#!/bin/bash

###queue
#PBS -q workq

### Requested number of nodes
#PBS -l nodes=2:ppn=20

### Requested computing time
#PBS -l walltime=18:00:00

### Account
#PBS -A TG-AST150015

### Job name
#PBS -N 'brownseds_agn'

### output and error logs
#PBS -o brownseds_agn_120.out
#PBS -e brownseds_agn_120.err

cd $PBS_O_WORKDIR

mpirun -np 40 -machinefile $PBS_NODEFILE \
python $APPS/prospector/scripts/prospector.py \
--param_file=$APPS/threedhst_bsfh/parameter_files/brownseds_agn/brownseds_agn_params_120.py
