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

source activate /well/rittscher/users/achatrian/.conda/envs/imageConvEnv
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:/well/rittscher/users/achatrian/cancer_phenotype/quant:${PYTHONPATH}"
COMMANDS="${1},--shuffle_images"
COMMANDS=$(tr ',' ' ' <<< ${COMMANDS})  # substitute commas with spaces
echo -e "Bioformat convert commands commands:\n ${COMMANDS}"
python /well/rittscher/users/achatrian/cancer_phenotype/aux/export/bioformat_convert_to_tiff.py ${COMMANDS}