#!/usr/bin/env bash

FILES=($1/*)

echo "Tarring directories in ${1} ..."
cd $1  # make .tar.gz files in source directory
COUNTER=0
for DIRPATH in "${FILES[@]}"; do
    if [[ -d "${DIRPATH}" ]]; then
        NAME=$(basename "${DIRPATH}")
        tar -czvf "${NAME}.tar.gz" ${DIRPATH}
        COUNTER=$((COUNTER+1))
    fi
done
echo "${COUNTER} directories were tarred and gzipped"