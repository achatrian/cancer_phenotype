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
LAUNCHDIR=/well/rittscher/users/achatrian/cancer_phenotype/launchscripts

if [[ -z $2 ]]
then
    SAVEDIR=/well/rittscher/projects/prostate-gland-phenotyping/WSI/data/dzi
    FILES=(/well/rittscher/projects/prostate-gland-phenotyping/WSI/*)
else
    SAVEDIR=$2/data/dzi
    FILES=($2/*)  # expand to all files / dirs in given data directory
fi
echo "Saving files in ${SAVEDIR}"
COUNTER=0
for SLIDEPATH in "${FILES[@]}"; do
    SLIDENAME=$(basename "${SLIDEPATH}") # get basename only
    # check if file has .ndpi or .svs format. If not, skip iteration
    if [[ $SLIDENAME == *.ndpi ]]
    then
        SLIDEID="${SLIDENAME%%.ndpi*}"
    elif [[ $SLIDENAME == *.svs ]]
    then
        SLIDEID="${SLIDENAME%%.svs*}"
    else
        # if iterating over dirs, look for images inside dirs
        for SUBPATH in ${SLIDEPATH}/*; do
            SUBNAME=$(basename "${SUBPATH}") # get basename only
            if [[ $SUBNAME == *.ndpi ]]
            then
                SLIDEID="${SUBNAME%%.ndpi*}"
                SLIDEPATH=${SUBPATH}
            elif [[ $SUBNAME == *.svs ]]
            then
                SLIDEID="${SUBNAME%%.svs*}"
                SLIDEPATH=${SUBPATH}
            else
                continue
            fi
        done  # NB: JOB IS LAUNCHED ONLY OR LAST IMAGE IN SUBDIR
    fi
    if [[ ! -z "$SLIDEID" ]]; then
        echo ${SLIDEID}
        COUNTER=$((COUNTER+1))
        SLIDECOMMANDS="${SLIDEPATH},${COMMANDS}"
        # option: -l h_rt=2:00:00  kills job within 2 hours of processing
        qsub -o "${LOGDIR}/o${JOBID}_${SLIDEID}" -e "${LOGDIR}/e${JOBID}_${SLIDEID}" -P rittscher.prjc -q long.qc -pe shmem 2 -l h_rt=2:00:00 ${LAUNCHDIR}/l_deepzoom.sh ${SLIDECOMMANDS} ${SAVEDIR}
    fi
done
echo -e "Deepzoom commands:\n ${COMMANDS}"
echo "Making deepzoom image for ${COUNTER} slides ..."