#!/usr/bin/env bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

DATA_DIR=/well/win/users/achatrian/ProstateCancer/Dataset
SAVE_DIR=/well/win/users/achatrian/ProstateCancer/Dataset/Results
CHKP_DIR=/well/win/users/achatrian/ProstateCancer/logs/2018_09_19_11_24_46/ckpt
SNAPSHOT=epoch_.411_loss_0.20396_acc_0.95000_dice_0.93000_lr_0.0000103566.pth
NF=$1
BS=$2

module load cuda/8.0
module load cudnn/6.0-8.0
source /users/win/achatrian/pytorch-v0.4.0-cuda8.0-py3.5-venv/bin/activate
cd /users/win/achatrian/ProstateCancer/phenotyping
echo "Start"
python /users/win/achatrian/ProstateCancer/phenotyping/extract_feats.py -d $DATA_DIR -chk $CHKP_DIR -snp $SNAPSHOT --gpu_ids 0 1 -sd $SAVE_DIR -nf $NF --batch_size $BS
echo "Done running script"
