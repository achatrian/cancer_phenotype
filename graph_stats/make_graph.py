
import os, sys
import re
import glob
import numpy as np

sys.path.append("../phenotyping")



def main(FLAGS):

    #Load large images containing whole glands
    gt_files = glob.glob(os.path.join(FLAGS.data_dir, '**','*_mask.png'), recursive=True) #for 1,1 patch
    img_files = [re.sub('mask', 'img', gtfile) for gtfile in gt_files]





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/ProstateCancer/Dataset")

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed: warnings.warn("Unparsed arguments")
    main(FLAGS)
