# get glands

import os, sys
import multiprocessing as mp
import imageio
import glob
from pathlib import Path
import re
import argparse
import warnings

import numpy as np
import cv2

sys.path.append("../segment")
from utils import check_mkdir

class InstanceSaver(mp.Process):
    def __init__(self, queue, id, dir, out_size, min_gland_area=10000, bb_margin=0.2):
        mp.Process.__init__(self, name='InstanceSaver')
        self.daemon = True #requred
        self.id = id
        self.queue = queue
        self.dir = Path(dir)
        self.out_size = out_size
        self.min_gland_area = min_gland_area
        self.bb_margin=bb_margin

    def run(self):
        count = 0
        while True:
            data = self.queue.get()
            if data is None:
                self.queue.task_done()
                break

            idx, img, gt, glandspath = data
            #Change colours to make gt visible
            gt[gt == 1] = 120
            gt[gt == 2] = 160
            gt[gt == 3] = 200
            gt[gt == 4] = 250
            gt2, contours, hierarchy = cv2.findContours(gt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for c_idx, contour in enumerate(contours):
                if self.min_gland_area and cv2.contourArea(contour) < self.min_gland_area:
                    continue  #skip instance if too small
                x,y,w,h = cv2.boundingRect(contour)  #get bounding box
                #Slightly increase bounding Box
                x -= round(w*self.bb_margin)
                y -= round(h*self.bb_margin)
                w += round(w*self.bb_margin)*2 #overall increases by size * margin percentage
                h += round(h*self.bb_margin)*2
                if x <= 0 or y <= 0 or x+w >= img.shape[1] or y+h >= img.shape[0]:
                    continue  #skip instance if gland touches walls (incomplete)
                gland_img, gland_gt = img[y:y+h, x:x+w], gt[y:y+h, x:x+w]

                #Resize so that image touches tile wall for long dimension - works for glands smaller or bigger than tile size
                #then pad short dimension till it touches wall -- long dimension is height
                if w >= h:
                    #align long dimension to height if it's not
                    gland_img = gland_img.transpose(1,0,2)
                    gland_gt = gland_gt.transpose(1,0)
                    x,y,w,h = y,x,h,w

                # Resize
                oh = self.out_size
                ow = round((oh / float(h)) * float(w))  #rounding is done towards the even choice (NB need even size for dcgan-vae implementation)
                top, bottom = 0,0
                delta_w = self.out_size - ow
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                gland_img = cv2.resize(gland_img, dsize=(ow, oh), interpolation=cv2.INTER_CUBIC)
                gland_gt = cv2.resize(gland_gt, dsize=(ow, oh), interpolation=cv2.INTER_CUBIC)
                gland_img = cv2.copyMakeBorder(gland_img, top, bottom, left, right, cv2.BORDER_REFLECT)
                gland_gt = cv2.copyMakeBorder(gland_gt, top, bottom, left, right, cv2.BORDER_REFLECT)  # pad short dimension

                imageio.imwrite(str(self.dir / glandspath / "gland_img_{}{}.png".format(idx, c_idx)), gland_img)
                imageio.imwrite(str(self.dir / glandspath / "gland_gt_{}{}.png".format(idx, c_idx)), gland_gt)
                count += 1
                if count % 100 == 0:
                    print("Process {} saved ~{} glands".format(self.id, count))

            self.queue.task_done()



def main(FLAGS):

    #Load large images containing whole glands
    gt_files = glob.glob(os.path.join(FLAGS.data_dir, '**','*_mask.png'), recursive=True) #for 1,1 patch
    img_files = [re.sub('mask', 'img', gtfile) for gtfile in gt_files]

    queue = mp.JoinableQueue(maxsize=2*FLAGS.workers)  #what determines good size?
    for i in range(FLAGS.workers):
        InstanceSaver(queue, i, dir=FLAGS.data_dir, out_size=FLAGS.image_size,
                      min_gland_area=FLAGS.min_gland_area, bb_margin=FLAGS.increase_bounding_box).start()

    for idx, (img_file, gt_file) in enumerate(zip(img_files, gt_files)):
        img = imageio.imread(img_file)
        gt = imageio.imread(gt_file)
        tilepath = Path(img_file).parents[0]
        glandspath = tilepath.parents[0]/"glands"/tilepath.name
        check_mkdir(glandspath.parents[0])
        check_mkdir(glandspath)
        queue.put((idx, img, gt, glandspath))

    for i in range(FLAGS.workers):
        queue.put(None)  # why?
    queue.join()

    print("Done !")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/cancer_phenotype/Dataset")
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to make gland data')
    parser.add_argument('--image_size', type=int, default=299)
    parser.add_argument('--min_gland_area', type=int, default=6000)
    parser.add_argument('-ib','--increase_bounding_box', type=float, default=0.1)
    mp.set_start_method('spawn')

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed: warnings.warn("Unparsed arguments")
    main(FLAGS)
