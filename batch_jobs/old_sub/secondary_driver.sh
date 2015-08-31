#!/bin/bash

RUNNAME=$1
NARRAY=$2
JOBNUM=$3

for ((  i = 1 ;  i <= $NARRAY;  i++  ))
do
  cmd='sbatch --dependency=afterany:'$JOBNUM'_'$i' --array='$i'-'$i' '$RUNNAME'_secondary.sh'
  $cmd
done