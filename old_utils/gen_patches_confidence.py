"""
Generate patches for training. This is part of creating a pipeline for the prostate project.
"""
import os
import openslide
import argparse

import numpy as np
import pandas as pd
import multiprocessing as mp

# from KS_lib import KSimage
# from KS_lib.prepare_data import routine
import imageio

class WriterWorkers(mp.Process):
    def __init__(self, slide_path, queue, gt_path):
        mp.Process.__init__(self, name = 'workers')
        self.daemon = True
        self._queue = queue
        self._slide_path = slide_path
        self._gt_path = gt_path

    def run(self):
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break

            wsi = openslide.open_slide(self._slide_path)

            location, savename, down_scale_level, patch_size = data

            img = wsi.read_region(location, down_scale_level, (patch_size, patch_size))
            img = np.array(img)[:, :, 0:3]

            imageio.imwrite(savename, img)

            self._queue.task_done()

def main():
    global args
    args = parser.parse_args()

    # read the paths of gt and wsi from a csv file
    csv_pointer = pd.read_csv(args.csv_file)
    gt_list = csv_pointer['gt']
    wsi_list = csv_pointer['wsi']

    # predefine folders for train/val and sub-folders for different classes
    save_dict = {'Stroma': os.path.join('Photos', args.folder, 'Stroma'),
                 'Tumour': os.path.join('Photos', args.folder, 'Tumour'),
                 'Benign': os.path.join('Photos', args.folder, 'Benign'),
                 'Lumen': os.path.join('Photos', args.folder, 'Lumen')}

    class_dict = {'Stroma': 2,
                  'Tumour': 4,
                  'Benign': 3,
                  'Lumen': 1}

    for key in save_dict.keys():
        if not os.path.exists(save_dict[key]):
            os.makedirs(save_dict[key])

    # loop through a pair of gt and wsi
    for gt_file, wsi_file in zip(gt_list, wsi_list):
        basename = os.path.basename(gt_file)
        basename = os.path.splitext(basename)[0]
        print(basename)

        # read the ground truth
        gt = imageio.imread(gt_file)

        # get a pointer to wsi
        wsi = openslide.OpenSlide(wsi_file)

        # # encode sobol sequence
        # X = sobol_seq.i4_sobol_generate(2, int(1e6))
        # X[:, 0] *= gt.shape[0]
        # X[:, 1] *= gt.shape[1]
        # X = X.astype(np.int)
        # mask_sobol = np.zeros(shape=gt.shape, dtype=np.bool)
        # for row in X:
        #     mask_sobol[row[0], row[1]] = True

        # get indices for each class
        for key in class_dict.keys():
            print(key)
            mask = gt == class_dict[key]
            # L, num = KSimage.bwlabel(mask)
            # L_dict = KSimage.label2idx(L)
            # del L_dict[0]

            # for idx in L_dict.keys():
            # indices = np.unravel_index(mask, mask.shape)

            indices = np.where(mask)[0] #added "[0]"
            order = np.random.permutation(len(indices[0]))

            # sample according to the square root of the area
            num_sample = np.int(np.sqrt(len(indices[0])))
            # num_sample = len(indices[0])

            shuffle_indices = []
            for index in indices:
                index = index[order]
                #index = index[:num_sample]
                shuffle_indices.append(index)

            counter = 0

            #########################################################
            num_workers = 8
            queue = mp.JoinableQueue(2 * num_workers)
            for i in range(num_workers):
                WriterWorkers(wsi_file, queue).start()
            #########################################################

            for x, y, z in zip(*shuffle_indices):
                # first the patch should overlap at least 80% of the ground truth
                # tmp = mask[x-8:x+8, y-8:y+8, :]

                # if (np.sum(tmp)/tmp.size) > 0.0:

                # the ground truth was generated at the reduction level = 3
                y0, x0 = y * (2 ** 3), x * (2 ** 3) # coordinate at level 0

                # find the patch size at level 0
                half_patch_size = int(args.patch_size/2.0)
                half_patch_size0 = half_patch_size*(2**args.down_scale_level)

                location = (y0 - half_patch_size0, x0 - half_patch_size0)
                savename = os.path.join(save_dict[key], basename + '_row_' + str(x) + '_col_' + str(y) + '.png')

                if not os.path.exists(savename):
                    queue.put((location, savename, args.down_scale_level, args.patch_size))
                    counter += 1
                # img = wsi.read_region(location, args.down_scale_level, (args.patch_size, args.patch_size))
                # img = np.array(img)[:, :, 0:3]
                #
                # savename = os.path.join(save_dict[key], basename + '_row_' + str(x) + '_col_' + str(y) + '.png')
                # KSimage.imwrite(img, savename)

                if counter > num_sample:
                    break

            ###############################################################
            for _i in range(num_workers):
                queue.put(None)
            queue.join()
            ###############################################################

        wsi.close()

def cut_tile(wsi_file, location, patch_size, savename, down_scale_level=3):
    wsi = openslide.OpenSlide(wsi_file)
    img = wsi.read_region(location, down_scale_level, (patch_size, patch_size))
    img = np.array(img)
    imageio.imwrite(savename, img)
    print("Saved tile {} at location {},{}".format(savename, location[0], location[1]))


if __name__ == '__main__':
    # define parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='./list_of_gt_and_wsi_val.csv')
    parser.add_argument('--folder', type=str, default='val')
    parser.add_argument('--down_scale_level', type=int, default=3) #labels are currently at level 3
    parser.add_argument('--export_folder', type=str, default='./GT_segmentation')
    parser.add_argument('--val_percentage', type=int, default=30)
    parser.add_argument('--patch_size', type=int, default=512)
    main()
