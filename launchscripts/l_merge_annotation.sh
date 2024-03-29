#!/usr/bin/env bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

# for when apply is launched by another script
module use -a /mgmt/modules/eb/modules/all #Make all modules available for loading
module load Anaconda3/5.1.0 #Load Anaconda

if [[ -z $2 ]]
then
    MAX_ITER=2
else
    MAX_ITER=$2
fi

source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:${PYTHONPATH}"
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype/base/utils:${PYTHONPATH}"
python /well/rittscher/users/achatrian/cancer_phenotype/base/utils/annotation_builder.py --annotation_path=$1 \
--max_iter=$MAX_ITER --append_merged_suffix
