#!/usr/bin/env bash

# Script takes --option=value strings separated by comma (,):
# E.g. --task=segment,--model=unet

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"



source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
COMMANDS=$1
echo -e "Apply commands:\n ${COMMANDS}"
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:${PYTHONPATH}"
shopt -s globstar
COUNTER=0
DATE=`date`
JOBID=$(tr ' ' '_' <<< ${DATE})  # replace spaces with underscores
JOBID=$(tr ':' '_' <<< ${JOBID})  # replace columns with underscores (or it breaks)
JOBID="${JOBID}_${JOB_ID}"  # in case multiple jobs are run in the same second, add counter id to differentiate between them
LOGDIR=/well/rittscher/users/achatrian/jobs_logs/apply_many_tile_based

array_contains () {
# first input is array and second is element
    local array="$1[@]"
    local seeking=$2
    CHECK=0
    for element in "${!array}"; do
        if [[ $element == $seeking ]]; then
            CHECK=1
            break
        fi
    done
}

IDS=()
for SLIDEPATH in /well/rittscher/users/achatrian/ProstateCancer/Dataset/{train,test}/*; do
    SLIDENAME=$(basename "${SLIDEPATH}") # get basename only
    SLIDEID="${SLIDENAME%%_TissueTrain_*}"
    SLIDECOMMANDS="${COMMANDS},--slide_id=${SLIDEID}"
    # echo "Applying UNET for ${SLIDEID}"
    array_contains IDS ${SLIDEID}
    if [[ ! -z SLIDEID ]] && [[ ${CHECK} -eq 0 ]]
    then
        touch $LOGDIR/"o${JOBID}_${SLIDEPATH}"
        touch $LOGDIR/"e${JOBID}_${SLIDEPATH}"
        qsub -o $LOGDIR/"o${JOBID}_${SLIDEPATH}" -e $LOGDIR/"e${JOBID}_${SLIDEPATH}" -P rittscher.prjc -q gpu8.q -l gpu=1 -l gputype=p100 ./l_apply.sh ${SLIDECOMMANDS}
        COUNTER=$((COUNTER+1))
        IDS+=(${SLIDEID})
    fi
done
echo "Applying network to ${COUNTER} slides ..."