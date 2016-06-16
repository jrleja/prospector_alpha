#!/bin/bash

# this is however many you want to submit, e.g. jobs 3-8
NSTART=$1
NARRAY=$2

for ((  i = $NSTART ;  i <= $NEND;  i++  ))
do
  cmd='qsub suball_'$i'.sh'
  $cmd
done