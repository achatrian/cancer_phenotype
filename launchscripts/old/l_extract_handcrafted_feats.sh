#!/usr/bin/env bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

DATA_DIR=/well/rittscher/users/achatrian/ProstateCancer/Dataset
SAVE_DIR=/well/rittscher/users/achatrian/ProstateCancer/Results/handcrafted
BS=$1
AUG=$2
WORKS=$3

echo "batch of " $BS " augmentation of " $AUG " and " $WORKS " processes "

module load cuda/9.0
module load cudnn/7.0-9.0
source /users/rittscher/achatrian/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate
cd /users/rittscher/achatrian/ProstateCancer
if [ $4 -eq 1 ]
then
    python /users/rittscher/achatrian/ProstateCancer/phenotyping/extract_handcrafted_feats_mp.py -d $DATA_DIR \
    -sd $SAVE_DIR --batch_size $BS --augment_num $AUG --workers $WORKS
else
    python /users/rittscher/achatrian/ProstateCancer/phenotyping/extract_handcrafted_feats.py -d $DATA_DIR \
    -sd $SAVE_DIR --batch_size $BS --augment_num $AUG
fi
echo "Finished at :"`date`