#!/bin/bash

###queue
#PBS -q workq

### Requested number of nodes
#PBS -l nodes=1:ppn=20

### Requested computing time
#PBS -l walltime=1:00:00

### Account
#PBS -A TG-AST150015

### Job name
#PBS -N 'np_mock'

### output and error logs
#PBS -o nonparametric_mock_1.out
#PBS -e nonparametric_mock_1.err

cd $PBS_O_WORKDIR

mpirun -np 20 -machinefile $PBS_NODEFILE \
python $APPS/prospector/scripts/prospector.py \
--nwalkers=38 \
--niter=20 \ 
--maxfev=50 \
--param_file=$APPS/threedhst_bsfh/parameter_files/nonparametric_mocks/nonparametric_mocks_params.py