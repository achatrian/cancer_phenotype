#get glands

import os, sys
import multiprocessing as mp
import imageio
import glob
from pathlib import Path
import re
import argparse
import warnings
from itertools import product

import numpy as np
import cv2

sys.path.append("../mymodel")
def check_mkdir(dir_name):
    try:
        os.mkdir(str(dir_name))
    except FileExistsError:
        pass

class InstanceSaver(mp.Process):
    """
    Groups together
    """
    def __init__(self, queue, id, dir, out_size, min_gland_area=6000, bb_margin=0.1):
        mp.Process.__init__(self, name='InstanceSaver')
        self.daemon = True #requred
        self.id = id
        self.queue = queue
        self.dir = Path(dir)
        self.max_out_size = out_size
        self.min_gland_area = min_gland_area
        self.bb_margin = bb_margin

    def run(self):
        count = 0
        while True:
            data = self.queue.get()
            if data is None:
                self.queue.task_done()
                break

            idx, img, gt, glandspath = data
            assert img.size
            assert gt.size

            # Resize as during training
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            gt = cv2.resize(gt, (0, 0), fx=0.5, fy=0.5)

            gt2, contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c_idx, contour in enumerate(contours):
                if self.min_gland_area and cv2.contourArea(contour) < self.min_gland_area:
                    continue  #skip instance if too small
                x,y,w,h = cv2.boundingRect(contour)  #get bounding box
                if x <= 0 or y <= 0 or x+w >= img.shape[1] or y+h >= img.shape[0]:
                    continue  #skip instance if gland touches walls (incomplete)

                # Slightly increase bounding Box
                x = max(x - round(w * self.bb_margin), 0)
                y = max(y - round(h * self.bb_margin), 0)
                w += min(round(w * self.bb_margin) * 2, img.shape[1] - x) # overall increases by size * margin percentage
                h += min(round(h * self.bb_margin) * 2, img.shape[0] - y)

                if w > self.max_out_size:
                    # For glands bigger than the desired tile size
                    tile_xs = list(range(x, x + w - self.max_out_size, self.max_out_size))
                    if tile_xs and x + w - self.max_out_size - tile_xs[-1] < 150:
                        del tile_xs[-1]  # delete last and replace with border if they are similar
                    tile_xs += [x + w - self.max_out_size] #  add extra tile to take end of gland
                else:
                    tile_xs = [x]

                if h > self.max_out_size:
                    tile_ys = list(range(y, y + h - self.max_out_size, self.max_out_size))
                    if tile_ys and y + h - self.max_out_size - tile_ys[-1] < 150:
                        del tile_ys[-1]  # delete last and replace with border if they are similar
                    tile_ys += [y + h - self.max_out_size]
                else:
                    tile_ys = [y]

                gt_temp = gt.copy()  # copy gt and zero everything around bounding box
                gt_temp[0:y, ...] = 0
                gt_temp[y+h:, ...] = 0
                gt_temp[:, 0:x] = 0
                gt_temp[:, x+w:] = 0

                # TODO: replace with contour based zeroing ?

                for xt, yt in product(tile_xs, tile_ys):
                    wt = min(w, self.max_out_size)
                    ht = min(h, self.max_out_size)
                    gland_img = img[yt: yt + ht, xt: xt + wt]
                    gland_gt = gt_temp[yt: yt + ht, xt: xt + wt]
                    assert gland_img.size  # must be non-empty
                    assert gland_gt.size # must be non-empty
                    imageio.imwrite(self.dir / glandspath / "gland_img_{}_({},{}).png".format(c_idx, xt, yt), gland_img)
                    imageio.imwrite(self.dir / glandspath / "gland_gt_{}_({},{}).png".format(c_idx, xt, yt), gland_gt)

                count += 1
                if count % 100 == 0:
                    print("Process {} saved ~{} glands".format(self.id, count))

            self.queue.task_done()



def main(FLAGS):

    #Load large images containing whole glands
    gt_files = glob.glob(os.path.join(FLAGS.data_dir, '**','*_mask.png'), recursive=True) # for 1,1 patch
    img_files = [re.sub('mask', 'img', gtfile) for gtfile in gt_files]

    queue = mp.JoinableQueue(maxsize=2*FLAGS.workers)  #what determines good size?
    for i in range(FLAGS.workers):
        InstanceSaver(queue, i, dir=FLAGS.data_dir, out_size=FLAGS.max_image_size,
                      min_gland_area=FLAGS.min_gland_area).start()

    for idx, (img_file, gt_file) in enumerate(zip(img_files, gt_files)):
        img = imageio.imread(img_file)
        gt = imageio.imread(gt_file)
        tilepath = Path(img_file).parents[0]
        glandspath = tilepath.parents[0]/"glands_full"/tilepath.name
        check_mkdir(glandspath.parents[0])
        check_mkdir(glandspath)
        queue.put((idx, img, gt, glandspath))

    for i in range(FLAGS.workers):
        queue.put(None)

    queue.join()
    # print("Generated {} glands (with minimum area {})".format(idx, FLAGS.min_gland_area))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/ProstateCancer/Dataset")
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to make gland data')
    parser.add_argument('--max_image_size', type=int, default=512)
    parser.add_argument('--min_gland_area', type=int, default=6000)
    mp.set_start_method('spawn')

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed: warnings.warn("Unparsed arguments")
    main(FLAGS)
