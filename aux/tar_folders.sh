#!/usr/bin/env bash

FILES=(${1}/*)

echo "Tarring directories in ${1} ..."
cd ${1} || exit  # make .tar.gz files in source directory
COUNTER=0
for DIRPATH in "${FILES[@]}"; do
    if [[ -d "${DIRPATH}" && ${DIRPATH} == *_files ]]; then
        NAME=$(basename "${DIRPATH}")
        echo "Tarring ${DIRPATH} ... "
        if [ -f "${NAME}.tar.gz" ]; then
          echo "File exists"
        else
          tar -czvf "${NAME}.tar.gz" ${DIRPATH}
        fi
        COUNTER=$((COUNTER+1))
    fi
done
echo "${COUNTER} directories were tarred and gzipped"