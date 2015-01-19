#!/bin/bash

NARRAY=$1
JOBNUM=$2

for ((  i = 1 ;  i <= $NARRAY;  i++  ))
do
  sbatch --dependency=afterany:$JOBNUM_$NARRAY --array=$NARRAY-$NARRAY nebon_secondary.sh
done