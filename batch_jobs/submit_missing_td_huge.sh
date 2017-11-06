#!/bin/bash
# loop over all jobs
# look for output
# if it doesn't exist, submit it
IDFILE=$APPS"/prospector_alpha/data/3dhst/td_huge.ids"

for value in {1..54418}
do
    OBJID=$(sed -n "${value}p" "$IDFILE")
    FIELD=${OBJID%_*}
    if ! ls $APPS/prospector_alpha/results/td_huge/$FIELD/${OBJID}_*_model 1> /dev/null 2>&1; then
        OUTFILE=$APPS/prospector_alpha/results/td_huge/${FIELD}/${OBJID}
        echo ${OBJID}, ${value}
        sbatch -o td_huge_${value}.out -e td_huge_${value}.err submit_one_td_huge.sh ${OBJID} ${OUTFILE}
    fi
done
wait
