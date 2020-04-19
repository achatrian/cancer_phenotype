import os
from glob import glob
from functools import reduce
import numpy as np
import imageio

DATA_DIR = "/well/rittscher/users/achatrian/cancer_phenotype/Dataset/03_nucleus/"
nuclei_inst_dirs = glob(DATA_DIR + "/train/*/masks_001") + glob(DATA_DIR + "/test/*/masks_001")

for inst_dir in nuclei_inst_dirs:
    image_files = [file_name for file_name in os.listdir(inst_dir) if os.path.isfile(os.path.join(inst_dir, file_name))]
    images = [np.array(imageio.imread(os.path.join(inst_dir, image_file))) for image_file in image_files]
    assert all(type(image) is np.ndarray for image in images)
    ground_truth = reduce(np.logical_or, images) * 255
    ground_truth = ground_truth.astype(np.uint8)
    gt_savename = image_files[0].split('_')[0:-1]  # get rid of numbering
    gt_savename = '_'.join(gt_savename) + "_gt.png"
    imageio.imwrite(os.path.join(inst_dir, gt_savename), ground_truth)







