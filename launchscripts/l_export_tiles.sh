#!/usr/bin/env bash
# script for launching commands to make tiles

i=0
REPO=/well/rittscher/users/achatrian/cancer_phenotype
for FILENAME in /well/rittscher/projects/TCGA_prostate/TCGA/*/*{.svs,.ndpi}
 do
    echo "slide ${i}: ${FILENAME}"
    qsub -q short.qc ${REPO}/aux/export/export_tiles.sh ${FILENAME}
    ((++i))
done
