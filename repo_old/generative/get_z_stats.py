import os
import sys
from numbers import Integral
import csv

import numpy as np
from scipy.stats import mode
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import cv2

import torch
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
    sys.path.append("/gpfs0/users/rittscher/achatrian/cancer_phenotype/pix2pix_cyclegan")
else:
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype")
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype/pix2pix_cyclegan")
from repo_old.generative.utils import TestOptions, segmap2img
from repo_old.phenotyping import VAE
from repo_old.phenotyping import GTDataset
from segment.utils import get_flags


def main(opts):
    exp_name=''
    opts.save_dir = opts.save_dir or os.path.join(opts.checkpoints_dir, "generated")
    try:
        os.mkdir(opts.save_dir)
    except FileExistsError:
        pass

    # For data parallel, default gpu must be set as first available one in node (pytorch rule)
    print("Running on gpus {}".format(opts.gpu_ids))

    if len(opts.gpu_ids) == 1:
        opts.gpu_ids = opts.gpu_ids[0]
    if torch.cuda.is_available():
        if isinstance(opts.gpu_ids, Integral):
            std_device = opts.gpu_ids
            torch.cuda.set_device(std_device)
            num_gpus = 1
            opts.gpu_ids = [opts.gpu_ids]
        else:
            std_device = opts.gpu_ids[0]
            torch.cuda.set_device(std_device)
            num_gpus = len(opts.gpu_ids)
            opts.batch_size -= opts.batch_size % len(opts.gpu_ids)  # ensure that there are enough examples per GPU
            opts.val_batch_size -= opts.batch_size % len(opts.gpu_ids)  # " "
    else:
        num_gpus = 0
        std_device = torch.device('cpu')

    # Maintain checkpoint dir / snapshot structure
    vae_snapshot = os.path.basename(opts.vaegan_file)
    vae_chkpt_dir = os.path.dirname(opts.vaegan_file)

    # Load VAE-GAN
    file_name = os.path.join(vae_chkpt_dir, os.path.basename(os.path.dirname(vae_chkpt_dir))[5:]) + ".txt"
    LOADEDFLAGS = get_flags(file_name)
    opts.num_filt_gen = LOADEDFLAGS.num_filt_gen
    opts.num_filt_discr = LOADEDFLAGS.num_filt_discr
    opts.num_lat_dim = LOADEDFLAGS.num_lat_dim
    opts.image_size = LOADEDFLAGS.image_size
    print("Loaded model settings - ngf: {}, ndf: {}, nz: {}".format(opts.num_filt_gen,
                                                                           opts.num_filt_discr,
                                                                           opts.num_lat_dim))
    num_channels = 4
    vae = VAE(opts.image_size, num_gpus, opts.num_filt_gen, opts.num_lat_dim, num_channels=num_channels)
    if torch.cuda.is_available():
        vae = vae.cuda()

    if vae_snapshot.endswith("D.pth"):
        vae_snapshot = vae_snapshot.replace("D.pth", "G.pth")
    if not isinstance(std_device, Integral) and std_device.type == "cpu":
        state_dict_G = torch.load(opts.vaegan_file, map_location=std_device)
    else:
        state_dict_G = torch.load(opts.vaegan_file)
    state_dict_G = {key[7:]: value for key, value in state_dict_G.items()}  # remove data_parallel's "module."
    vae.load_state_dict(state_dict_G)

    print("vaegan: {}".format(vae_snapshot))

    vae.eval()

    dataset = GTDataset(opts.dataroot, opts.phase, augment=False, tile_size=128)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=opts.shuffle,  num_workers=4)

    # Pass through pipeline
    # data is dict {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
    mse = MSELoss(size_average=True)  # use to take the 2 most different images
    mus_dir = os.path.join(opts.save_dir, opts.phase + "_repr", "mus")
    logvars_dir = os.path.join(opts.save_dir, opts.phase + "_repr", "logvars")
    try:
        os.makedirs(mus_dir), os.makedirs(logvars_dir)
    except FileExistsError:
        pass
    real_gland_areas_ck5pos, rec_gland_areas_ck5pos = [], []
    real_gland_areas_ck5neg, rec_gland_areas_ck5neg = [], []
    mse = MSELoss(size_average=True)
    with tqdm(total=len(dataloader)) as pbar:
        for i, (segmaps, labels, paths) in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                if i >= opts.max_img_num:
                    break
                if torch.cuda.is_available():
                    segmaps = segmaps.cuda()
                # Extract representation and encoded from network
                (mus, logvars), reconstructed, generated = vae(segmaps)
                differences = [mse(rec, segmap) for rec, segmap in zip(reconstructed, segmaps)]
            mus = mus.cpu().numpy().transpose(0, 2, 3, 1)
            logvars = logvars.cpu().numpy().transpose(0, 2, 3, 1)
            for path, mu, logvar, diff in zip(paths, mus, logvars, differences):
                name = os.path.basename(path)
                np.save(os.path.join(mus_dir, "mu_{:.4f}_".format(diff) + name), mu)
                np.save(os.path.join(logvars_dir, "logvar_{:.4f}_".format(diff) + name), logvar)
            recs = segmap2img(reconstructed, return_tensors=False)[..., 0]  # all 3 channels are identical
            gts = segmap2img(segmaps, return_tensors=False)[..., 0]

            for s in range(gts.shape[0]):
                gt = gts[s].copy()
                rec = recs[s].copy()
                _0, contours_gts, _1 = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                _0, contours_recs, _1 = cv2.findContours(rec, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Store real and fake gland areas
                for cnt in contours_gts:
                    x, y, w, h = cv2.boundingRect(cnt)
                    gland = gts[s, y:y+h, x:x+w]
                    gland_area = cv2.contourArea(cnt)
                    if gland_area > 300:
                        class_mask = np.logical_and(gland > 0, np.logical_not(np.isclose(gland, 250, atol=20)))
                        if np.isclose(mode(gland[class_mask], axis=None,)[0], 160, atol=20):
                            real_gland_areas_ck5neg.append(gland_area)
                        elif np.isclose(mode(gland[class_mask], axis=None,)[0], 200, atol=20):
                            real_gland_areas_ck5pos.append(gland_area)
                for cnt in contours_recs:
                    x, y, w, h = cv2.boundingRect(cnt)
                    gland = gts[s, y:y+h, x:x+w]
                    gland_area = cv2.contourArea(cnt)
                    if gland_area > 300:
                        class_mask = np.logical_and(gland > 0, np.logical_not(np.isclose(gland, 250, atol=20)))
                        if np.isclose(mode(gland[class_mask], axis=None,)[0], 160, atol=20):
                            rec_gland_areas_ck5neg.append(gland_area)
                        elif np.isclose(mode(gland[class_mask], axis=None,)[0], 200, atol=20):
                            rec_gland_areas_ck5pos.append(gland_area)
            pbar.update(1)

    with open(os.path.join(opts.save_dir, opts.phase + "_areas.tsv"), 'w') as area_file:
        wr = csv.writer(area_file, delimiter="\t")
        wr.writerow(["REAL_CK5-"] + real_gland_areas_ck5neg)
        wr.writerow(["REAL_CK5+"] + real_gland_areas_ck5pos)
        wr.writerow(["REC_CK5-"] + rec_gland_areas_ck5neg)
        wr.writerow(["REC_CK5+"] + rec_gland_areas_ck5pos)

    print("Done!")


if __name__ == "__main__":
    opts = TestOptions().parse()
    main(opts)




















