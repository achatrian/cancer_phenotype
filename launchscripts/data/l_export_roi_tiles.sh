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


source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
COMMANDS=$(tr ',' ' ' <<< $1)  # substitute commas with spaces
echo -e "Export ROI tiles commands:\n ${COMMANDS}"
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:${PYTHONPATH}"
python3 /well/rittscher/users/achatrian/cancer_phenotype/data/roi_tile_exporter.py ${COMMANDS}