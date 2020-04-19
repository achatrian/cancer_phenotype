#!/usr/bin/env bash

if [[ -n "$1" ]]
then
    GT=$1
else
    GT='gland_gt*.png' # glob for ground truth images
fi

if [[ -n "$2" ]]
then
    IMG=$2
else
    IMG='gland_img*.png' # glob for histology images
fi

if [[ -n "$3" ]]
then
    TO_REPLACE=$3
else
    TO_REPLACE=gt
fi

if [[ -n "$3" ]]
then
    TO_REPLACE=$3
else
    TO_REPLACE=gt
fi

if [[ -n "$4" ]]
then
    REPLACE_WITH=$4
else
    REPLACE_WITH=img
fi

if [[ -n "$5" ]]
then
    COPY_FROM=$5
else
    COPY_FROM=glands_full/
fi

if [[ -n "$6" ]]
then
    COPY_TO=$6
else
    COPY_TO=../gan_full/
fi

echo "Copying from $COPY_FROM to $COPY_TO"
echo "Using \"$GT\" to find A images, \"$IMG\" to find B images"
for dir in "train" "val" "test" ; do
    echo "Making " $dir
    # Copy glands into A and B folders
    mkdir -p $COPY_TO
    mkdir -p $COPY_TO/A
    mkdir -p $COPY_TO/B
    mkdir -p $COPY_TO/A/$dir/
    mkdir -p $COPY_TO/B/$dir/
    find $COPY_FROM/$dir -type f -name $GT -exec cp --remove-destination '{}' $COPY_TO/A/$dir/ ';'
    find $COPY_FROM/$dir -type f -name $IMG -exec cp --remove-destination '{}' $COPY_TO/B/$dir/ ';'
    cd $COPY_TO/A/$dir
    echo "Replace $TO_REPLACE with $REPLACE_WITH in $COPY_TO/A/$dir"
    rename $TO_REPLACE $REPLACE_WITH *
    # rename does not work on normal file with -s option on ubuntu
done

if [[ "`hostname`" = "local" ]]
then
    cd ~/Documents/Repositories/ProstateCancer/pytorch-CycleGAN-and-pix2pix/
else
    source ~/pytorch-0.4.1-cuda9.0-py3.5.2-local-install/bin/activate
    cd ~/ProstateCancer/pix2pix_cyclegan
fi
echo "Merging"
python3 datasets/combine_A_and_B.py --fold_A $COPY_TO/A \
--fold_B $COPY_TO/B --fold_AB $COPY_TO/AB
echo "Done"