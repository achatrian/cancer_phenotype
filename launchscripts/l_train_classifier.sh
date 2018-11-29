#!/usr/bin/env bash

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
source /users/rittscher/achatrian/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate
cd /users/rittscher/achatrian/ProstateCancer/ck5
echo "Start"
python3 train_classifier.py --gpu_ids 0 1 --batch_size 140