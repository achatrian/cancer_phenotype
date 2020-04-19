#!/bin/bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

DATA_DIR=/well/rittscher/users/achatrian/ProstateCancer/Dataset
NF=$1
BS=$2
AUG=$3
WORKS=$4
AUG_DIR=$5
CHKPT_DIR=$6
SNPS=$7


module load cuda/9.0
module load cudnn/7.0-9.0
source /users/rittscher/achatrian/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate

# Load model if given location and name

if [ -z "$CHKPT_DIR" ] || [ -z "$SNPS" ]  # if either variable is empty
then
    echo "Start training"
    python /users/rittscher/achatrian/ProstateCancer/mymodel/train_net.py -nf $1 --batch_size=$2 --augment $3 \
    --workers=$4 --gpu_ids 0 1 --augment_dir=$AUG_DIR
else
    echo "Restart training"
    python /users/rittscher/achatrian/ProstateCancer/mymodel/train_net.py -nf $1 --batch_size=$2 --augment $3 \
    --augment_dir=$AUG_DIR --workers=$WORKS --gpu_ids 0 1 --chkpt_dir=$CHKPT_DIR --snapshot=$SNPS
fi
echo "Done!"