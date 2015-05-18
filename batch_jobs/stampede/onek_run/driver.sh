#!/bin/bash

RUNNAME=$1
NARRAY=$2
JOBNUM=$3

for ((  i = 1 ;  i <= $NARRAY;  i++  ))
do
  cmd='sbatch onek_run_suball_'$i'.sh'
  $cmd
done