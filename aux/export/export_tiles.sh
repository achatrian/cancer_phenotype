#!/usr/bin/env bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

REPO=/well/rittscher/users/achatrian/cancer_phenotype
#Locations where conda environments are stored:
#Make all modules available for loading
module use -a /mgmt/modules/eb/modules/all
#Load Anaconda
module load Anaconda3/5.1.0
module load cuda/9.0
module load cudnn/7.0-9.0
source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
python ${REPO}/aux/export/export_tiles.py ${1}





