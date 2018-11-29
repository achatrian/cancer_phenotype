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

echo "batch of " $BS " augmentation of " $AUG " and " $WORKS " processes "

module load cuda/9.0
module load cudnn/7.0-9.0
source /users/rittscher/achatrian/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate
cd /users/rittscher/achatrian/ProstateCancer/phenotyping
echo "SCRIPT STARTS"
python3 train_net_vae_gt.py --gpu_ids 0 1 --batch_size=$BS --val_batch_size=$BS -nz=1920 -ndf 23 -ngf 22 --full_glands=y --image_size=256
