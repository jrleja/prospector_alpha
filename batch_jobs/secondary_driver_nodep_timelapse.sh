#!/bin/bash

RUNNAME=$1
NARRAY=$2
DELTAT='2m'

for ((  i = 1 ;  i <= $NARRAY;  i++  ))
do
  cmd='sbatch --array='$i'-'$i' '$RUNNAME'_secondary.sh'
  $cmd
  sleep $DELTAT
done