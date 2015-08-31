#!/bin/bash
# script to submit multiple jobs
# over a specified amount of time

NJOBS=108
JOBS_PER_BATCH=18
NBATCH=$(($NJOBS/$JOBS_PER_BATCH))
DELTAT='90m'

echo 'Submitting '$NJOBS' jobs, with '$JOBS_PER_BATCH' jobs per batch in '$NBATCH' batches.'
echo 'Waiting '$DELTAT' between submissions'

for ((  i = 1 ;  i <= $NBATCH;  i++  ))
do
	
	JOBSTART=$((($i-1)*$JOBS_PER_BATCH+1))
	JOBEND=$(($i*$JOBS_PER_BATCH))
	
	sbatch --array=$JOBSTART-$JOBEND submit_job_all.sh

	echo 'submitting batch set '$i', jobs '$JOBSTART'-'$JOBEND
	
	sleep $DELTAT
	
done