#!/usr/bin/env bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

source activate /well/rittscher/users/achatrian/.conda/envs/imageConvEnv

SOURCEDIR=$1
TARGETDIR=$2
echo "Source: ${SOURCEDIR}"
echo "Target: ${TARGETDIR}"

if [ ! -d "$TARGETDIR" ]; then
    mkdir -p "${TARGETDIR}"
fi

COUNTER=0
for TIFFPATH in ${SOURCEDIR}/*.tiff; do
    NAME=$(basename "${TIFFPATH}")
    echo "${NAME}"
    ID="${NAME%%.tiff*}"
    PYRAMIDPATH="${TARGETDIR}/${ID}.tiff"
    if [[ ${NAME} == *.tiff && ! ( -f ${PYRAMIDPATH} ) ]]  # save only if file does not already exist
    then
        COUNTER=$((COUNTER+1))
        echo "Saving ${PYRAMIDPATH} ..."
        vips tiffsave ${TIFFPATH} ${PYRAMIDPATH} --compression=jpeg --Q=90 --tile --tile-width=256 --tile-height=256 --pyramid --bigtiff
    fi
done