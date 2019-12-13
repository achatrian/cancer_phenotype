#!/usr/bin/env bash

echo "************************************************************************************"
echo "SGE job ID: "$JOB_ID
echo "SGE task ID: "$SGE_TASK_ID
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "************************************************************************************"

# CONVERTS MICROSCOPY IMAGES IN BIOFORMATS COMPATIBLE FORMATS TO TIFFS.
# IF THE SOURCE IMAGE HAS A PYRAMIDAL STRUCTURE, THE TARGET IMAGE WILL ALSO HAVE ONE (to test)

module load LibTIFF/4.0.9-GCCcore-7.3.0  # for vips?
source activate /well/rittscher/users/achatrian/.conda/envs/pyenvclone

SOURCEDIR=$1
TARGETDIR=$2
SUFFIX=$3
if [ -z "$SOURCEDIR" ]
then
  exit 1
fi
if [ -z "$TARGETDIR" ]
then
  exit 1
fi
if [ -z "$SUFFIX" ]
then
  exit 1
fi
echo "Source: ${SOURCEDIR}"
echo "Target: ${TARGETDIR}"
mkdir -p ${TARGETDIR}

COUNTER=0
for IMAGEPATH in ${SOURCEDIR}/*/*.${SUFFIX}; do  # dir/image_dir/image
    NAME=$(basename "${IMAGEPATH}")
    ID="${NAME%%.svs*}"
    TIFFPATH="${TARGETDIR}/${ID}.tiff"
    echo "Converting ${TIFFPATH} ..."
    if [[ ! (-f ${TIFFPATH}) ]]  # save only if file does not already exist
    then
        COUNTER=$((COUNTER+1))
        bfconvert -bigtiff ${IMAGEPATH} ${TIFFPATH}
    fi
done