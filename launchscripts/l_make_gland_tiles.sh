#!/bin/bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

DATA_DIR=/gpfs0/well/rittscher/users/achatrian/ProstateCancer/Dataset

module load cuda/8.0
module load cudnn/6.0-8.0
source /users/rittscher/achatrian/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate
cd ~/ProstateCancer/mymodel
python /users/rittscher/achatrian/ProstateCancer/phenotyping/make_gland_tiles.py -d $DATA_DIR
echo "Done running script"
