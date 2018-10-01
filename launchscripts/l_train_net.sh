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
MODEL_FILENAME=/users/win/achatrian/ProstateCancer/logs/2018_08_21_15_20_31/ckpt/epoch_.101_loss_0.49436_acc_0.83000_dice_0.79000_lr_0.0001000000.pth

module load cuda/8.0
module load cudnn/6.0-8.0
source /users/win/achatrian/pytorch-v0.4.0-cuda8.0-py3.5-venv/bin/activate
python /users/win/achatrian/ProstateCancer/mymodel/get_hard_examples.py -mf $MODEL_FILENAME -dd $DATA_DIR -sw=y --batch_size=20
