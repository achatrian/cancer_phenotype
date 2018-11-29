import os
import sys
from numbers import Integral
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import MSELoss


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
from repo_old.generative.utils import save_images, TestOptions, img2segmap, segmap2img
from repo_old.phenotyping import VAE
from pix2pix_cyclegan.models import create_model
from pix2pix_cyclegan.data.aligned_dataset import AlignedDataset
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
            #std_device = torch.device("cuda:{}".format(opts.gpu_ids))
            std_device = opts.gpu_ids
            torch.cuda.set_device(std_device)
            num_gpus = 1
            opts.gpu_ids = [opts.gpu_ids]
        else:
            #std_device = torch.device("cuda:{}".format(opts.gpu_ids[0]))
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
    print("Loaded settings of repo_old model - ngf: {}, ndf: {}, nz: {}".format(opts.num_filt_gen,
                                                                           opts.num_filt_discr,
                                                                           opts.num_lat_dim))
    num_channels = 4
    #discr = Discriminator(opts.image_size, num_gpus, opts.num_filt_discr,
    #                      num_channels)  # (self, image_size, ngpu, num_filt_discr, num_channels=3):
    vae = VAE(128, num_gpus, opts.num_filt_gen, opts.num_lat_dim, num_channels=num_channels)
    if torch.cuda.is_available():
        vae = vae.cuda()

    if vae_snapshot.endswith("G.pth"):
        if not isinstance(std_device, Integral) and std_device.type == "cpu":
            state_dict_G = torch.load(opts.vaegan_file, map_location=std_device)
            #state_dict_D = torch.load(opts.vaegan_file.replace("G.pth", "D.pth"), map_location=std_device)
        else:
            state_dict_G = torch.load(opts.vaegan_file)
    #elif vae_snapshot.endswith("D.pth"):
    #    # don't need discriminator here
    #    state_dict_D = torch.load(opts.vaegan_file, map_location=std_device)
    #    state_dict_G = torch.load(opts.vaegan_file.replace("D.pth", "G.pth"), map_location=std_device)
    state_dict_G = {key[7:]: value for key, value in state_dict_G.items()}  # remove data_parallel's "module."
    #state_dict_D = {key[7:]: value for key, value in state_dict_D.items()}
    vae.load_state_dict(state_dict_G)
    #discr.load_state_dict(state_dict_D)

    print("vaegan: {}".format(vae_snapshot))

    #discr.eval()
    vae.eval()

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
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opts.eval:
        pix2pix.eval()

    datasets = []
    for phase in ["train"]:
        opts.phase = phase
        dataset = AlignedDataset()
        dataset.initialize(opts)
        datasets.append(dataset)
    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=opts.shuffle, num_workers=4)

    # Pass through pipeline
    # data is dict {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
    mse = MSELoss(size_average=True)  # use to take the 2 most different images
    img_count = 0
    whole_data_run_count = 0
    while img_count < opts.max_img_num:
        with tqdm(total=len(dataloader)) as pbar:
            for i, data in tqdm(enumerate(dataloader)):
                if i >= opts.max_img_num:
                    break
                A_real, B_real = data['A'], data['B']
                seg_map_real, labels_real = img2segmap(A_real, return_tensors=True, size=128)
                if torch.cuda.is_available():
                    seg_map_real = seg_map_real.cuda()
                reconstructed_store, generated_store = [], []
                mse_rec_old, mse_gen_old, mse_rec, mse_gen = 0, 0, 0, 0
                # Take the most dissimilar samples
                for s in range(opts.num_samples):
                    encoded, reconstructed, generated = vae(seg_map_real)
                    if s > 1:
                        mse_rec = mse(reconstructed_store[0], reconstructed)
                        mse_gen = mse(generated_store[0], generated)
                        if mse_rec > mse_rec_old:
                            reconstructed_store.pop()
                            reconstructed_store.append(reconstructed.detach())
                        if mse_gen > mse_gen_old:
                            generated_store.pop()
                            generated_store.append(generated.detach())
                    else:
                        reconstructed_store.append(reconstructed.detach())
                        generated_store.append(generated.detach())

                # Need to mimic structure of data:
                seg_map_real = segmap2img(seg_map_real, return_tensors=True)
                reconstructed = segmap2img(torch.cat(reconstructed_store, 0), return_tensors=True)
                generated = segmap2img(torch.cat(generated_store, 0), return_tensors=True)
                new_data = data.copy()
                new_data['A'] = torch.cat((seg_map_real, reconstructed, generated), dim=0)
                new_data['B'] = B_real.repeat(3 if opts.num_samples == 1 else 6, 1, 1, 1)
                origin_len = seg_map_real.shape[0]
                # Add paths to save
                new_data['A_paths'] += [path[:-4] + "_rec1.png" for path in new_data['A_paths']]
                new_data['B_paths'] += [path[:-4] + "_rec1.png" for path in new_data['B_paths']]
                if len(reconstructed_store) == 2:
                    new_data['A_paths'] += [path[:-4] + "_rec2.png" for path in new_data['A_paths'][:origin_len]]
                    new_data['B_paths'] += [path[:-4] + "_rec2.png" for path in new_data['B_paths'][:origin_len]]
                new_data['A_paths'] += [path[:-4] + "_gen1.png" for path in new_data['A_paths'][:origin_len]]
                new_data['B_paths'] += [path[:-4] + "_gen1.png" for path in new_data['B_paths'][:origin_len]]
                if len(generated_store) == 2:
                    new_data['A_paths'] += [path[:-4] + "_gen2.png" for path in new_data['A_paths'][:origin_len]]
                    new_data['B_paths'] += [path[:-4] + "_gen2.png" for path in new_data['B_paths'][:origin_len]]

                pix2pix.set_input(new_data)
                pix2pix.test()
                visuals = pix2pix.get_current_visuals()
                img_path = pix2pix.get_image_paths()
                img_count += reconstructed.shape[0] + generated.shape[0]
                save_images(visuals, img_path, opts.save_dir)
                pbar.update(1)
                if img_count >= opts.max_img_num:
                    break
            tqdm.write("[run: {}] image count: {}".format(whole_data_run_count, img_count))


if __name__ == "__main__":
    opts = TestOptions().parse()
    main(opts)




















