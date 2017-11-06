#!/bin/bash
# loop over all jobs
# look for output
# if it doesn't exist, submit it
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_huge.ids"

for value in {1..54418}
do
    OBJID=$(sed -n "${n}p" "$IDFILE")
    FIELD=${OBJID%_*}
    mod=${OBJID//[ ]/*}
    if ! ls $APPS/prospector_alpha/results/td_huge/$field/*$mod*h5 1> /dev/null 2>&1; then
        sbatch -o td_huge_${value}.out -e td_huge_${value}.err submit_one_td_huge.sh ${OBJID} ${FIELD} $value
    fi
done
wait
