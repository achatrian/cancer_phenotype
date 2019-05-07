#!/usr/bin/env bash

# Script takes --option=value strings separated by comma (,):
# E.g. --task=segment,--model=unet
# Works on both ndpi and svs files

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

module load cuda/9.0
module load cudnn/7.0-9.0
source activate /well/rittscher/users/achatprostate-gland-phenotypingrian/.conda/envs/pyenvclone
COMMANDS=$1
echo -e "Apply commands:\n ${COMMANDS}"
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/generate:${PYTHONPATH}"
shopt -s globstar
DATE=`date`
JOBID=$(tr ' ' '_' <<< ${DATE})  # replace spaces with underscores
JOBID=$(tr ':' '_' <<< ${JOBID})  # replace columns with underscores (or it breaks)
JOBID="${JOBID}_${JOB_ID}"  # in case multiple jobs are run in the same second, add counter id to differentiate between them
LOGDIR=/well/rittscher/users/achatrian/jobs_logs/apply_many

if [[ -z $2 ]]
then
    FILES=(/well/rittscher/projects/prostate-gland-phenotyping/WSI/*)
else
    FILES=($2/*)  # expand to all files / dirs in given data directory
fi

COUNTER=0
for SLIDEPATH in "${FILES[@]}"; do
    SLIDENAME=$(basename "${SLIDEPATH}") # get basename only
    # check if file has .ndpi or .svs format. If not, skip iteration
    if [[ $SLIDENAME == *.ndpi ]]
    then
        SLIDEID="${SLIDENAME%%.ndpi*}"
        COUNTER=$((COUNTER+1))
    elif [[ $SLIDENAME == *.svs ]]
    then
        SLIDEID="${SLIDENAME%%.svs*}"
        COUNTER=$((COUNTER+1))
    else
        continue
    fi
    SLIDECOMMANDS="${COMMANDS},--slide_id=${SLIDEID}"
    qsub -o "${LOGDIR}/o${JOBID}_${SLIDEID}" -e "${LOGDIR}/e${JOBID}_${SLIDEID}" -P rittscher.prjc -q gpu8.q -l gpu=1 -l gputype=p100 ./l_apply.sh ${SLIDECOMMANDS}
done
echo "Applying network to ${COUNTER} slides ..."