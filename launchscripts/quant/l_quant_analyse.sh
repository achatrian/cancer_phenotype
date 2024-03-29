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
COMMANDS=$(tr ',' ' ' <<< $1)  # substitute commas with spaces
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:/well/rittscher/users/achatrian/cancer_phenotype/quant:${PYTHONPATH}"
echo -e "Analyse commands:\n ${COMMANDS}"
echo -e "${CONDA_PREFIX}"
python  /well/rittscher/users/achatrian/cancer_phenotype/quant/analyse.py ${COMMANDS}