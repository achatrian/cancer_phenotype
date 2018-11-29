#!/bin/bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

FEATS=$1
LABELS=$2
SAVE_DIR=$3

echo "Features file: " $FEATS
echo "Label file: " $FEATS
echo "Saving at: " $SAVE_DIR

source ~/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate
cd ~/ProstateCancer/phenotyping
python /users/rittscher/achatrian/ProstateCancer/phenotyping/cluster_feats.py -f $FEATS -lf $LABELS -sd $SAVE_DIR
echo "Done running script"