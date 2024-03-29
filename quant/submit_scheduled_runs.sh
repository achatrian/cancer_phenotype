#!/usr/bin/env bash

# Script takes --option=value strings separated by comma (,):
# E.g. --task=segment,--model=unet
# Works on both ndpi and svs files

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
export PYTHONPATH="/well/rittscher/users/achatrian/cancer_phenotype:/well/rittscher/users/achatrian/cancer_phenotype/base:/well/rittscher/users/achatrian/cancer_phenotype/segment:/well/rittscher/users/achatrian/cancer_phenotype/phenotype:/well/rittscher/users/achatrian/cancer_phenotype/encode:${PYTHONPATH}"
shopt -s globstar
DATE=`date`
JOBID=$(tr ' ' '_' <<< ${DATE})  # replace spaces with underscores
JOBID=$(tr ':' '_' <<< ${JOBID})  # replace columns with underscores (or it breaks)
JOBID="${JOBID}_${JOB_ID}"  # in case multiple jobs are run in the same second, add counter id to differentiate between them
LOGDIR="/well/rittscher/users/achatrian/jobs_logs/run_experiments"

RUNS_PARAMETERS_FILE=$1
COMMON_PARAMETERS=$2
echo -e "Shared run commands:\n ${COMMON_PARAMETERS}"
cd /well/rittscher/users/achatrian/cancer_phenotype/launchscripts/quant || exit
COUNTER=0
while IFS="" read -r RUN_PARAMETERS || [ -n "$RUN_PARAMETERS" ]
do
  RUN_COMMANDS="${COMMON_PARAMETERS},${RUN_PARAMETERS}"
  printf 'Submitting %s\n' "$RUN_COMMANDS"
  # FIXME logging file specification gives error: Unable to run job: ERROR! two files are specified for the same host.
  qsub -o "${LOGDIR}/run_${COUNTER}.o" -e "${LOGDIR}/run_${COUNTER}.e" -P rittscher.prjc -q long.qc -pe shmem 4 ./l_run_mcl_experiment.sh "${RUN_COMMANDS}"
  COUNTER=$((COUNTER+1))
done < "${RUNS_PARAMETERS_FILE}"
printf "\n"
echo "${COUNTER} experiment runs were launched with shared parameters: ${COMMON_PARAMETERS}"


