#!/usr/bin/env bash

# Script takes --option=value strings separated by comma (,):
# E.g. --task=tcga,--experiment=mcl

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/well/rittscher/projects/base/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/well/rittscher/projects/base/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/well/rittscher/projects/base/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/well/rittscher/projects/base/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export CONDA_ENVS_PATH=/well/rittscher/users/achatrian/.conda/envs
conda activate pyenvclone
COMMANDS=$(tr ',' ' ' <<< $1)  # substitute commas with spaces
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:/well/rittscher/users/achatrian/cancer_phenotype/quant:${PYTHONPATH}"
echo -e "Analyse commands:\n ${COMMANDS}"
python  /well/rittscher/users/achatrian/cancer_phenotype/quant/run.py ${COMMANDS}