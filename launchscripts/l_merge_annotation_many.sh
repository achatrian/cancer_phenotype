#!/usr/bin/env bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

# for when apply is launched by another script
module use -a /mgmt/modules/eb/modules/all #Make all modules available for loading
module load Anaconda3/5.1.0 #Load Anaconda
DATE=`date`
JOBID=$(tr ' ' '_' <<< ${DATE})  # replace spaces with underscores
JOBID="${JOBID}_${JOB_ID}"  # in case multiple jobs are run in the same second, add counter id to differentiate between them
LOGDIR=/well/rittscher/users/achatrian/jobs_logs/merge_annotation_many

if [[ -z $2 ]]
then
    MAX_ITER=2
else
    MAX_ITER=$2
fi

COUNTER=0
for ANNOTATIONPATH in $1/*; do
    if [[ ${ANNOTATIONPATH: -5} == ".json" ]]
    then
        qsub -o $LOGDIR/"o${JOBID}_${ANNOTATIONPATH}" -e $LOGDIR/"e${JOBID}_${ANNOTATIONPATH}" -P rittscher.prjc -q long.qc ./l_merge_annotation.sh $ANNOTATIONPATH $MAX_ITER
        COUNTER=$((COUNTER+1))
    fi
done
echo "Merging ${COUNTER} annotations ..."