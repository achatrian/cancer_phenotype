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

module load cuda/9.0
module load cudnn/7.0-9.0
source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
COMMANDS=$1
TCGA=$2
echo -e "Apply commands:\n ${COMMANDS}"
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/generate:${PYTHONPATH}"
shopt -s globstar
COUNTER=0

# working on TCGA
for SLIDEPATH in /well/rittscher/projects/TCGA_prostate/TCGA/*/*.svs; do
    if [[ -f "$i" ]]
    then
        break
    fi
    # echo "Applying UNET for ${SLIDEID}"
    qsub -P rittscher.prjc -q gpu8.q -l gpu=1 -l gputype=p100 ./l_test.sh ${COMMANDS}
    COUNTER=$((COUNTER+1))
done
echo "Applying network to ${COUNTER} slides ..."