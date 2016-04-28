#!/bin/bash

# this is however many you want to submit
NARRAY=$1

for ((  i = 1 ;  i <= $NARRAY;  i++  ))
do
  cmd='qsub suball_'$i'.sh'
  $cmd
done