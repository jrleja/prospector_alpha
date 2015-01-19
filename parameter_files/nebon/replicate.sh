#!/bin/bash
# script to replicate parameter file
# for multiple galaxy IDs
# set location of ID file, base parameter file
IDFILE=$APPS"/threedhst_bsfh/data/COSMOS_testsamp.ids"
PARAMFOLDER=$APPS"/threedhst_bsfh/parameter_files/nebon/"
PARAMBASE="halpha_selected_nebon_params"
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
  
  # find line with "'objname': '9'", replace 9 with $i
  sed -ie "s/\'objname\':\'15431\'/\'objname\':\'$LINE\'/g" $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT
  rm $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT"e"
done