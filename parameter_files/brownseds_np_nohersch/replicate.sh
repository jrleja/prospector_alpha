#!/bin/bash
# script to replicate parameter file
# for multiple galaxy IDs
# set location of ID file, base parameter file
IDFILE=$APPS"/prospector_alpha/data/herschel_names.txt"
PARAMFOLDER=$APPS"/prospector_alpha/parameter_files/brownseds_np_nohersch/"
PARAMBASE="brownseds_np_nohersch_params"
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
  sed -ie "s/\'objname\':\'Arp 256 N\'/\'objname\':\'$LINE\'/g" $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT
  rm $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT"e"
done