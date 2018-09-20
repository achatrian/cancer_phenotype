#!/bin/bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

DATA_DIR=/well/win/users/achatrian/ProstateCancer/Dataset

module load cuda/8.0
module load cudnn/6.0-8.0
source /users/win/achatrian/pytorch-v0.4.0-cuda8.0-py3.5-venv/bin/activate
cd ~/ProstateCancer/mymodel
python /users/win/achatrian/ProstateCancer/phenotyping/make_gland_tiles_full.py -d $DATA_DIR
echo "Done running script"
