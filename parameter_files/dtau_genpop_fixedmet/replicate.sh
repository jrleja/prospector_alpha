#!/bin/bash
# script to replicate parameter file
# for multiple galaxy IDs
# set location of ID file, base parameter file
IDFILE="met.dat"
PARAMBASE="dtau_genpop_fixedmet_params"
PARAMEXT=".py"

echo 'Base parameter file:'$PARAMBASE

# get number of IDs
NIDS=13

# loop
for ((  i = 1 ;  i <= $NIDS;  i++  ))
do
  # create new file
  cp $PARAMBASE$PARAMEXT $PARAMBASE"_$i"$PARAMEXT
  
  # read id
  LINE=$(sed -n "${i}p" "$IDFILE")
  echo $LINE
  
  : 

  # find line with "'objname': '15431'", replace 9 with $i
  sed -ie "s/\'init\': -1.0/\'init\': $LINE/g" $PARAMBASE"_$i"$PARAMEXT
  rm $PARAMBASE"_$i"$PARAMEXT"e"
done