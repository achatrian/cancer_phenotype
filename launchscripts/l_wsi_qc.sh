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
LOGDIR=/well/rittscher/users/achatrian/jobs_logs
OUTPUTLOG=${LOGDIR}/wsi_qc/"$(date)_o.txt"
ERRORLOG=${LOGDIR}/wsi_qc/"$(date)_e.txt"

# for when apply is launched by another script
module use -a /mgmt/modules/eb/modules/all #Make all modules available for loading
module load Anaconda3/5.1.0 #Load Anaconda



source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
COMMANDS=$(tr ',' ' ' <<< $1)
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:${PYTHONPATH}"
python /well/rittscher/users/achatrian/cancer_phenotype/aux/quality_control/wsi_qc.py ${COMMANDS} 1>${OUTPUTLOG} 2> ${ERRORLOG}