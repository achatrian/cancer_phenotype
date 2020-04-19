#!/usr/bin/env bash
# script for launching commands to make tiles
REPO=/well/rittscher/users/achatrian/cancer_phenotype
if [[ -z $1 ]]
then
i=0
for FILENAME in /well/rittscher/projects/TCGA_prostate/TCGA/*/*{.svs,.ndpi}
 do
    echo "slide ${i}: ${FILENAME}"
    qsub -q long.qc ${REPO}/aux/export/export_tiles.sh ${FILENAME}
    ((++i))
done
else
    echo "Building resolution info for dataset ..."
    module use -a /mgmt/modules/eb/modules/all #Make all modules available for loading
    module load Anaconda3/5.1.0 #Load Anaconda

    module load cuda/9.0
    module load cudnn/7.0-9.0
    source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
    python ${REPO}/aux/quality_control/build_resolution_data.py
fi
# run again with 1 as argument to merge the resolution data

