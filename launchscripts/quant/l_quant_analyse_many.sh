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

source /well/rittscher/users/achatrian/.bashrc
source /well/rittscher/users/achatrian/.bash_profile
conda activate pyenv
  export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:/well/rittscher/users/achatrian/cancer_phenotype/quant:${PYTHONPATH}"
echo -e "Commands:\n ${COMMANDS}"
echo -e "${CONDA_PREFIX}"
shopt -s globstar
DATE=`date`
JOBID=$(tr ' ' '_' <<< ${DATE})  # replace spaces with underscores
JOBID=$(tr ':' '_' <<< ${JOBID})  # replace columns with underscores (or it breaks)
JOBID="${JOBID}_${JOB_ID}"  # in case multiple jobs are run in the same second, add counter id to differentiate between them
LOGDIR=/well/rittscher/users/achatrian/jobs_logs/quant_analyse
mkdir -p $LOGDIR
cd /well/rittscher/users/achatrian/cancer_phenotype/launchscripts/quant || exit

COMMANDS=$1
if [[ -z $2 ]]
then
    FILES=(/well/rittscher/projects/TCGA_prostate/TCGA/data/annotations/combined_mpp1.0_normal_nuclei_circles/*)
else
    FILES=($2/*)  # expand to all files / dirs in given data directory
fi

COUNTER=0
for SLIDEPATH in "${FILES[@]}"; do
    SLIDENAME=$(basename "${SLIDEPATH}") # get basename only
    # check whether file has .json format. If not, skip iteration
    if [[ $SLIDENAME == *.json ]]
    then
        SLIDEID="${SLIDENAME%%.json*}"
    else
        # if iterating over dirs, look for images inside dirs
        for SUBPATH in ${SLIDEPATH}/*; do
            SUBNAME=$(basename "${SUBPATH}") # get basename only
            if [[ $SUBNAME == *.json ]]
            then
                SLIDEID="${SUBNAME%%.json*}"
            else
                continue
            fi
        done  # NB: JOB IS LAUNCHED ONLY FOR LAST IMAGE IN SUBDIR
    fi
    if [[ ! -z "$SLIDEID" ]]; then
        echo ${SLIDEID}
        COUNTER=$((COUNTER+1))
        SLIDECOMMANDS="${COMMANDS},--slide_id=${SLIDEID}"
#        echo $SLIDECOMMANDS
        qsub -o "${LOGDIR}/o${JOBID}_${SLIDEID}" -e "${LOGDIR}/e${JOBID}_${SLIDEID}" -q long.qc -pe shmem 4 ./l_quant_analyse.sh ${SLIDECOMMANDS}
    fi
done