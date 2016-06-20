#!/bin/bash
# script to replicate parameter file
# for multiple galaxy IDs
# set location of ID file, base parameter file
IDFILE=$APPS"/threedhst_bsfh/data/nonparametric_mocks.ids"
PARAMBASE="suball"
PARAMEXT=".sh"

echo 'ID file:'$IDFILE

# get number of IDs
NIDS=$(wc -l < "$IDFILE")
echo $NIDS

# loop
for ((  i = 1 ;  i <= $NIDS;  i++  ))
do
  # create new file
  cp $PARAMBASE$PARAMEXT $PARAMBASE"_$i"$PARAMEXT

  # find line with SLURM_ARRAY_TASK_ID, replace with i
  sed -ie "s/_1/_$i/g" $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT
  rm $PARAMBASE"_$i"$PARAMEXT"e"

  echo $i
done