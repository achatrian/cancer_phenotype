#!/usr/bin/env bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

DATA_DIR=/well/rittscher/users/achatrian/ProstateCancer/Dataset/pix2pix/AB
SAVE_DIR=/well/rittscher/users/achatrian/ProstateCancer/Dataset/pix2pix/generated
BS=$1

DR=/well/rittscher/users/achatrian/ProstateCancer/Dataset/pix2pix/AB
CHKPT=/well/rittscher/users/achatrian/ProstateCancer/logs/pix2pix_big
VAE=/well/rittscher/users/achatrian/ProstateCancer/logs/gtvae2018_10_11_18_38_22/ckpt/epoch_.3991_rec_loss_398118.46354_enc_loss_0.27605_dec_loss_0.10399_dis_loss_0.37233G.pth
GPU='0,1'

module load cuda/9.0
module load cudnn/7.0-9.0
source /users/rittscher/achatrian/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate
cd /users/rittscher/achatrian/ProstateCancer/generative
echo "SCRIPT STARTS"
python3 gen_gland_examples.py --dataroot $DR --checkpoints_dir $CHKPT --vaegan_file $VAE --model=pix2pix \
--ngf=180 --ndf=130 --batch_size=$BS --num_samples=1 --max_img_num 6000 --save_dir $SAVE_DIR --shuffle