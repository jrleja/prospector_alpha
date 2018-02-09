#!/bin/bash
# loop over all jobs
# look for output
# if it doesn't exist, submit it
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_new.ids"

for value in {1..9999}
do
    OBJID=$(sed -n "${value}p" "$IDFILE")
    FIELD=${OBJID%_*}
    if ! ls $APPS/prospector_alpha/results/td_new/${OBJID}_*h5 1> /dev/null 2>&1; then
        OUTFILE=$APPS/prospector_alpha/results/td_new/${OBJID}
        echo ${OBJID}, ${value}
        sbatch -o td_new_${value}.out -e td_new_${value}.err submit_one_td_new.sh ${OBJID} ${OUTFILE}
    fi
done
wait
