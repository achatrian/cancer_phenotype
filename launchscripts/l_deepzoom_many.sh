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
source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
COMMANDS=$1
echo -e "Deepzoom commands:\n ${COMMANDS}"
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:${PYTHONPATH}"
shopt -s globstar
DATE=`date`
JOBID=$(tr ' ' '_' <<< ${DATE})  # replace spaces with underscores
JOBID=$(tr ':' '_' <<< ${JOBID})  # replace columns with underscores (or it breaks)
JOBID="${JOBID}_${JOB_ID}"  # in case multiple jobs are run in the same second, add counter id to differentiate between them
LOGDIR=/well/rittscher/users/achatrian/jobs_logs/deepzoom_many
mkdir -p ${LOGDIR}

if [[ -z $2 ]]
then
    SAVEDIR=/well/rittscher/projects/prostate-gland-phenotyping/WSI/data/dzi
    FILES=(/well/rittscher/projects/prostate-gland-phenotyping/WSI/*)
else
    SAVEDIR=$2/data/dzi
    FILES=($2/*)  # expand to all files / dirs in given data directory
fi


COUNTER=0
for PATH_ in "${FILES[@]}"; do
    NAME=$(basename "${PATH_}") # get basename only
    # check if file has .ndpi or .svs format. If not, skip iteration
    if [[ $NAME == *.ndpi ]]
    then
        SLIDEPATH=${PATH_}
        COUNTER=$((COUNTER+1))
    elif [[ $NAME == *.svs ]]
    then
        SLIDEPATH=${PATH_}
        COUNTER=$((COUNTER+1))
    else
        # if iterating over dirs, look for images inside dirs
        for SUBPATH in ${PATH_}/*; do
            SUBNAME=$(basename "${SUBPATH}") # get basename only
            if [[ $SUBNAME == *.ndpi ]]
            then
                SLIDEPATH=${SUBPATH}
                COUNTER=$((COUNTER+1))
            elif [[ $SUBNAME == *.svs ]]
            then
                SLIDEPATH=${SUBPATH}
                COUNTER=$((COUNTER+1))
            else
                continue
            fi
        done  # NB: JOB IS LAUNCHED ONLY OR LAST IMAGE IN SUBDIR
    fi
    echo "${SLIDEPATH}"
    SLIDECOMMANDS="${SLIDEPATH},${COMMANDS}"
    qsub -o "${LOGDIR}/o${JOBID}_${NAME}" -e "${LOGDIR}/e${JOBID}_${NAME}" -P rittscher.prjc -q long.qc -pe shmem 2 ./l_deepzoom.sh ${SLIDECOMMANDS} ${SAVEDIR}
done
echo "Making deepzoom image for ${COUNTER} slides ..."