#!/bin/bash

###queue
#PBS -q workq

### Requested number of nodes
#PBS -l nodes=2:ppn=20

### Requested computing time
#PBS -l walltime=4:00:00

### Account
#PBS -A TG-AST150015

### Job name
#PBS -N 'np_mock'

### output and error logs
#PBS -o np_mocks_smooth_77.out
#PBS -e np_mocks_smooth_77.err

cd $PBS_O_WORKDIR

mpirun -np 40 -machinefile $PBS_NODEFILE \
python $APPS/prospector/scripts/prospector.py \
--param_file=$APPS/threedhst_bsfh/parameter_files/np_mocks_smooth/np_mocks_smooth_params_77.py
