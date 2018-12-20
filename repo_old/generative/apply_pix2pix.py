import os
import sys
from PIL import Image
import numpy as np
import torch
import imageio
from torchvision.transforms import ToTensor, Normalize
import cv2

def on_cluster():
    import socket, re
    hostname = socket.gethostname()
    match1 = re.search("jalapeno(\w\w)?.fmrib.ox.ac.uk", hostname)
    match2 = re.search("cuda(\w\w)?.fmrib.ox.ac.uk", hostname)
    match3 = re.search("login(\w\w)?.cluster", hostname)
    match4 = re.search("gpu(\w\w)?", hostname)
    match5 = re.search("compG(\w\w\w)?", hostname)
    match6 = re.search("rescomp(\w)?", hostname)
    return bool(match1 or match2 or match3 or match4 or match5)
if on_cluster():
    sys.path.append("/gpfs0/users/rittscher/achatrian/cancer_phenotype")
    sys.path.append("/gpfs0/users/rittscher/achatrian/cancer_phenotype/generate")
else:
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype")
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype/generate")
from repo_old.generative.utils import save_images, TestOptions
from generate.models import create_model

def make_input(segmaps):
    segmaps_temp = []
    for segmap in segmaps:
        segmap = Image.fromarray(segmap).convert('RGB')
        segmap = ToTensor()(segmap).float()
        segmap = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(segmap)
        segmaps_temp.append(segmap)
    segmaps = torch.cat(segmaps_temp)


opts = TestOptions().parse()
eps_path = "/Volumes/A-CH-EXDISK/Projects/Results/test_repr/eps_sample.npz"
with open(eps_path, 'rb') as eps_file:
    recs = np.load(eps_file)
    recs0 = recs['recs0']
    recs1 = recs['recs1']

old_path = "/Users/andreachatrian/Desktop/3991/2_input.png"
new_path = "/Users/andreachatrian/Desktop/3991/2_rec.png"
old_img = imageio.imread(old_path)
new_img = imageio.imread(new_path)
old_img = cv2.resize(old_img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
new_img = cv2.resize(new_img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)


# hard-code some parameters for test
opts.dataset_mode = "aligned"
opts.num_threads = 1  # test code only supports num_threads = 1
#opts.batch_size = 1  # test code only supports batch_size = 1
opts.serial_batches = True  # no shuffle
opts.no_flip = True  # no flip
opts.display_id = -1  # no visdom display
opts.num_test = None
pix2pix = create_model(opts)
pix2pix.setup(opts)
pix2pix.visual_names.remove("real_B")
rec0_path = ["rec0_{}".format(i) for i in range(recs0.shape[0])]
rec1_path = ["rec1_{}".format(i) for i in range(recs1.shape[0])]
save_dir = os.path.dirname(eps_path)
recs0 = torch.from_numpy(recs0).float()
recs1 = torch.from_numpy(recs1).float()

def feed_save(img_t, img_path, save_dir, pix2pix):
    data = {'A': img_t, 'A_paths': img_path, 'B': img_t, 'B_paths': img_path}
    pix2pix.set_input(data)
    pix2pix.test()
    visuals = pix2pix.get_current_visuals()
    img_path = pix2pix.get_image_paths()
    save_images(visuals, image_dir=save_dir, image_paths=img_path)

feed_save(recs0, rec0_path, save_dir, pix2pix)
feed_save(recs1, rec1_path, save_dir, pix2pix)
#oldnew_img = make_input([old_img, new_img])
#feed_save(oldnew_img, [old_path, new_path], save_dir, pix2pix)
