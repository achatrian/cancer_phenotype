#!/usr/bin/env bash

# Script takes --option=value strings separated by comma (,):
# E.g. --task=segmemt,--model=unet

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

# for when test is launched by another script
module use -a /mgmt/modules/eb/modules/all #Make all modules available for loading
module load Anaconda3/5.1.0 #Load Anaconda

module load cuda/9.0
module load cudnn/7.0-9.0
source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
COMMANDS=$(tr ',' ' ' <<< $1)  # substitute commas with spaces
echo -e "Train commands:\n ${COMMANDS}"
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/generate:${PYTHONPATH}"
python  /well/rittscher/users/achatrian/cancer_phenotype/base/test.py ${COMMANDS}