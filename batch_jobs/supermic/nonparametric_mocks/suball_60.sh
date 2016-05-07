#!/bin/bash

###queue
#PBS -q workq

### Requested number of nodes
#PBS -l nodes=2:ppn=20

### Requested computing time
#PBS -l walltime=6:00:00

### Account
#PBS -A TG-AST150015

### Job name
#PBS -N 'np_mock'

### output and error logs
#PBS -o nonparametric_mock_60.out
#PBS -e nonparametric_mock_60.err

cd $PBS_O_WORKDIR

mpirun -np 40 -machinefile $PBS_NODEFILE \
python $APPS/prospector/scripts/prospector.py \
--param_file=$APPS/threedhst_bsfh/parameter_files/nonparametric_mocks/nonparametric_mocks_params_60.py
