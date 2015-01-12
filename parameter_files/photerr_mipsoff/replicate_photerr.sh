#!/bin/bash
# script to replicate parameter file
# for multiple galaxy IDs
# set location of ID file, base parameter file
PARAMFOLDER=$APPS"/threedhst_bsfh/parameter_files/photerr_mipsoff/"
IDFILE="photerr.txt"
PARAMBASE="photerr_mipsoff_params"
PARAMEXT=".py"

echo 'ID file:'$IDFILE
echo 'Base parameter file:'$PARAMBASE

# get number of IDs
NIDS=$(wc -l < "$IDFILE")
echo $NIDS

# loop
for ((  i = 1 ;  i <= $NIDS;  i++  ))
do
  # create new file
  cp $PARAMFOLDER$PARAMBASE$PARAMEXT $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT

  # read id
  LINE=$(sed -n "${i}p" "$IDFILE")
  echo $LINE

  # find and replace minimum error
  sed -ie "s/\'min_error\': 0.02/\'min_error\': $LINE/g" $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT
  rm $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT"e"
done