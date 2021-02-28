#!/usr/bin/env bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"



source /users/rittscher/achatrian/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate
python3 ~/ProstateCancer/pytorch-CycleGAN-and-pix2pix/train.py --dataroot /well/rittscher/users/achatrian/ProstateCancer/Dataset/pix2pix --print_freq=40 \
--gpu_ids 0 1 --update_html_freq=100 --checkpoints_dir /well/rittscher/users/achatrian/ProstateCancer/logs/pix2pix \
--batch_size=30 --model pix2pix --loadSize 256 --display_freq=100 --continue_train