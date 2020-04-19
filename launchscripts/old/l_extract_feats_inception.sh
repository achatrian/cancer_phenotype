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
SAVE_DIR=/well/rittscher/users/achatrian/ProstateCancer/Results/augstudy
#CKM=/well/rittscher/users/achatrian/cancer_phenotype/logs/ck5_2018_10_01_16_12_53/ckpt/epoch_.91_loss_0.26689_acc_1.00000_dice_1.00000_lr_0.0000407000.pth
BS=$1
AUG=$2
CKM=$3


module load cuda/9.0
module load cudnn/7.0-9.0
source /users/rittscher/achatrian/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate
cd /users/rittscher/achatrian/ProstateCancer/phenotyping
echo "Start"
python /users/rittscher/achatrian/ProstateCancer/phenotyping/extract_feats_inception.py -d $DATA_DIR --ck_model $CKM --gpu_ids 0 1 \
-sd $SAVE_DIR --batch_size $BS --augment_num $AUG
echo "Done running script"