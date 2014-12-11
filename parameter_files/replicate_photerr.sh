#!/bin/bash
# script to replicate parameter files
# set location of ID file, base parameter file
IDFILE=$APPS"threedhst_bsfh/parameter_files/photerr.txt"
PARAMFOLDER=$APPS"threedhst_bsfh/parameter_files/"
PARAMBASE="photerr_param"
PARAMEXT=".py"

# number of points
NPOINTS=15

# loop
for ((  i = 1 ;  i <= $NPOINTS;  i++  ))
do
  
  # create new file
  #cp $PARAMFOLDER$PARAMBASE$PARAMEXT $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT

  # read id
  LINE=$(sed -n "${i}p" "$IDFILE")
  echo $LINE

  # find line with "'objname': '9'", replace 9 with $i
  #photerr=`echo 'l('$((20*($i+5)))')' | bc -l`
  photerr=$((10**$LINE))
  echo $photerr
  #sed -ie "s/\'objname\':\'9\'/\'objname\':\'$LINE\'/g" $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT
  #rm $PARAMFOLDER$PARAMBASE"_$i"$PARAMEXT"e"
done