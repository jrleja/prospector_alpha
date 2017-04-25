#!/bin/bash
# script to replicate parameter file
# for multiple galaxy IDs
# set location of ID file, base parameter file
runname=$1
IDFILE=$APPS"/threedhst_bsfh/data/brownseds_data/photometry/namelist.txt"

# get number of IDs
NIDS=$(wc -l < "$IDFILE")

# loop
for ((  i = 1 ;  i <= $NIDS;  i++  ))
do
  # read id
  LINE=$(sed -n "${i}p" "$IDFILE")
  mod=${LINE//[ ]/*}
  if ! ls $runname/*$mod* 1> /dev/null 2>&1; then
    echo $i" not done!"
  fi

done