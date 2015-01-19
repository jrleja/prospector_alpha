#!/bin/bash

RUNNAME=$1
NARRAY=$2
JOBNUM=$3

for ((  i = 1 ;  i <= $NARRAY;  i++  ))
do
  sbatch --dependency=afterany:$JOBNUM_$NARRAY --array=$i-$i $RUNNAME_secondary.sh
done