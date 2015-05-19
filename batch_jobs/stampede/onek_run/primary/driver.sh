#!/bin/bash

RUNNAME=$1
NARRAY=$2

for ((  i = 1 ;  i <= $NARRAY;  i++  ))
do
  cmd='sbatch '$RUNNAME'_suball_'$i'.sh'
  $cmd
done