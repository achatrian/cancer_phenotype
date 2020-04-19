#!/usr/bin/env bash

# create path list for visualization of images in a QuPath project

source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone
shopt -s globstar
DATE=`date`

if [[ -z $2 ]]
then
    SAVEDIR=/well/rittscher/projects/TCGA_prostate/TCGA/data
    FILES=(/well/rittscher/projects/TCGA_prostate/TCGA/*)
else
    SAVEDIR=$2/data
    FILES=($2/*)  # expand to all files / dirs in given data directory
fi

touch ${SAVEDIR}/image_list.txt

COUNTER=0
for PATH_ in "${FILES[@]}"; do
    NAME=$(basename "${PATH_}") # get basename only
    # check if file has .ndpi or .svs format. If not, skip iteration
    if [[ $NAME == *.ndpi ]]
    then
        SLIDEPATH=${PATH_}
        COUNTER=$((COUNTER+1))
    elif [[ $NAME == *.svs ]]
    then
        SLIDEPATH=${PATH_}
        COUNTER=$((COUNTER+1))
    else
        # if iterating over dirs, look for images inside dirs
        for SUBPATH in ${PATH_}/*; do
            SUBNAME=$(basename "${SUBPATH}") # get basename only
            if [[ $SUBNAME == *.ndpi ]]
            then
                SLIDEPATH=${SUBPATH}
                COUNTER=$((COUNTER+1))
            elif [[ $SUBNAME == *.svs ]]
            then
                SLIDEPATH=${SUBPATH}
                COUNTER=$((COUNTER+1))
            else
                continue
            fi
        done
    fi
    echo ${SLIDEPATH//well\/rittscher/mnt\/rescomp} >> ${SAVEDIR}/image_list.txt
done
echo "${COUNTER} paths were written to file"