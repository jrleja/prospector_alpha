#!/bin/bash
# script to replicate parameter file
# for multiple galaxy IDs
# set location of ID file, base parameter file
runname=$1
IDFILE=$APPS"/prospector_alpha/data/3dhst/td.ids"

# get number of IDs
NIDS=$(wc -l < "$IDFILE")

# loop
for ((  i = 1 ;  i <= $NIDS;  i++  ))
do
  # read id
  LINE=$(sed -n "${i}p" "$IDFILE")
  mod=${LINE//[ ]/*}
  if ! ls $APPS/prospector_alpha/results/$runname/*$mod*model 1> /dev/null 2>&1; then
    sbatch --array=$i-$i td_suball.sh
  fi

done