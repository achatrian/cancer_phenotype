import os, sys
import argparse
from warnings import warn
from pathlib import Path
from numbers import Integral

import numpy as np
from scipy.stats import kurtosis, skew, mode, entropy
import torch
from torch import cuda, load
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import cv2
from tqdm import tqdm
from imageio import imwrite
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from imgaug import augmenters as iaa
import imgaug as ia
from torch.nn.functional import avg_pool2d, interpolate
from itertools import chain
import mahotas as mts

sys.path.append("../ck5")
from ck5.inception import Inception

from skimage.measure import regionprops

from gland_dataset import GlandPatchDataset


dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


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
    sys.path.append(os.path.expanduser('~') + '/ProstateCancer')
else:
    sys.path.append(os.path.expanduser('~') + '/Documents/Repositories/ProstateCancer')
from mymodel.utils import on_cluster, get_time_stamp, check_mkdir, str2bool, \
    evaluate_multilabel, colorize, AverageMeter, get_flags
from mymodel.models import UNet1, UNet2, UNet3, UNet4

exp_name = ''


class FeatureExtractor(nn.Module):
    """
    Class for extracting features from desired submodules in neural network.
    """

    # FIXME Doesn't work for upsampling branch of UNET yet
    def __init__(self, submodule, extracted_layers, layers=None):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers.copy()
        if layers == "fill":
            layers = list(self.submodule._modules.keys())  # try to use keys (and hope they are in right order)
        self.layers = layers or ['input_block', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'enc6', 'center',
                                 'dec6', 'dec5', 'dec4', 'dec3', 'dec2', 'dec1']
        for layer in layers:
            if layer.startswith('dec') and layer != 'dec6':
                self.extracted_layers.append("enc" + str(int(layers[-1] - 1)))
            else:
                self.extracted_layers.append("center")


    def forward(self, x):
        to_extract = self.extracted_layers.copy()
        outputs = []
        # for name, module in self.submodule._modules.items():
        for n, name in enumerate(self.layers):
            module = self.submodule._modules[name]
            if not to_extract:
                continue
            x = self.change_x(x, name, outputs)
            x = module(x)
            if name in to_extract:
                outputs += [x]
                to_extract.remove(name)
        return outputs

    def change_x(self, x, name, outputs):
        if name in ['fc']:
            x = x.view(x.size(0), -1)  # for networks that end with fc classifier
        elif name.startswith('dec'):
            # decoders use input from previous encoders
            enc_name = "enc" + str(int(name[-1] - 1)) if name != "dec6" else "center"
            if enc_name in self.extracted_layers:
                enc_idx = self.extracted_layers.index(enc_name)
                enc_x = outputs[enc_idx]
                x = torch.cat([x, interpolate(outputs[enc_idx], x.size()[2:], mode='bilinear')], 1)

        return x


def get_gland_bb(gt):
    """
    :param gt:
    :return: bb for largest area in image, assumed to be gland in dataset
    """
    gt = gt.squeeze().astype(np.uint8) * 255
    gt2, contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    gland_contour = contours[areas.index(max(areas))]
    x, y, w, h = cv2.boundingRect(gland_contour)
    return x, y, w, h


def summary_net_features(gts, feature_maps):
    """
    :param gts:
    :param feature_maps:
    :return: gland feature vectors from path feature maps

    Features are mean, median, std, skew, kurtosis, entropy (from histogram)
    """

    glands_features = []
    empty_gts = []
    for n, gt in enumerate(gts):
        gland_feature_maps = [fm[n, ...] for fm in feature_maps]  # get feature maps for one gland
        # From feature maps:
        gl_features = []
        for fm in gland_feature_maps:
            gt_small = cv2.resize(gt[:, :, 0], fm.shape[-2:], interpolation=cv2.INTER_NEAREST)
            gl_features.append(fm[:, gt_small > 0])  # only take values overlapping with glands
        # Take spatial stats:
        for i, fm in enumerate(gl_features):
            # fm is 2D, first dim is channel and second is flattened spatial values
            try:
                fm_stats = [fm.mean(axis=1), np.median(fm, axis=1), fm.std(axis=1), fm.max(axis=1), fm.min(axis=1)]
            except ValueError:
                fm = np.zeros((gl_features[i].shape[0], 1))
                fm_stats = [fm.mean(axis=1), np.median(fm, axis=1), fm.std(axis=1), fm.max(axis=1), fm.min(axis=1)]
            fm_stats = np.array(fm_stats).T  # channels x features
            fm_stats = fm_stats.astype(np.float32)  # for reducing size and checking later
            gl_features[i] = fm_stats.flatten()
        gl_features = np.concatenate(gl_features)
        # FIXME numpy version still doesn't work -- fix handling of bad examples ??? recheck if works
        glands_features.append(gl_features)

    glands_features = np.array(glands_features)
    return glands_features


def get_stats(n, feature_maps, downsampled_gts, down_sizes):
    """
    :param fm:
    :param downsampled_gts:
    :return:
    used for pooling
    """

    all_gland_stats = []
    # Extract features for each gland
    gland_feature_maps = [fm[n, ...] for fm in feature_maps]  # get feature maps for one gland
    for i, fm in enumerate(gland_feature_maps):
        gtd = downsampled_gts[down_sizes.index(fm.shape[-1])][n, ...]
        try:
            fm_gland = fm[:, gtd.squeeze()]
            means = fm_gland.mean(dim=1)
        except RuntimeError:
            # when gt is empty -- fail fast and fill with zeros
            fm_gland = torch.zeros(fm.shape[0], 1).cuda()
            means = fm_gland.mean(dim=1)
        medians = fm_gland.mean(dim=1)
        varns = fm_gland.var(dim=1)
        maxs = fm_gland.max(dim=1)[0]
        mins = fm_gland.min(dim=1)[0]
        norms = fm_gland.norm(p=1, dim=1)
        stats = torch.cat([means, medians, varns, maxs, mins, norms], dim=0)
        all_gland_stats.append(stats)
    all_gland_stats = torch.cat(all_gland_stats, dim=0)
    return all_gland_stats


intrplt = lambda gts, sl: interpolate(gts, size=(sl, sl), mode='bilinear')
def summary_net_features_torch(gts, feature_maps):
    """
    :param gts: NCHW
    :param feature_maps: list of tensors [NCHW, NCHW, ...]
    :downsamplings: list of downsizes to calculates gts for
    :return: gland feature vectors from path feature maps

    Features are mean, median, var
    """

    glands_features = []
    fm_side_lens = sorted(list(set((fm.shape[3] for fm in feature_maps))), reverse=True)
    gt_sizes = [gts.shape[3]] + fm_side_lens
    matched_gts = [gts.byte()]
    matched_gts += [intrplt(gts, sl).byte() for sl in fm_side_lens]

    # TODO what about the upsampling path

    glands_features = []
    for n in range(gts.shape[0]):
       glands_features.append(get_stats(n, feature_maps, matched_gts, gt_sizes))
    glands_features = torch.stack(glands_features, dim=0)
    return glands_features


# Make objects for feature extraction
orb = cv2.ORB_create()  # ORB features

kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 5):
        for frequency in (0.05, 0.25):
            # Gabor patches
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


def feature_extraction(imgs, gts):
    """
    :param imgs:
    :param gts:
    :return:

    Extract features:
    SIFT
    ORB > SIFT
    SHAPE
    FILTER BANK RESPONSE
    """

    features = []
    for n, (img, gt) in enumerate(zip(imgs, gts)):
        if not gt.any():
            gt = np.ones(gt.shape, dtype=np.bool)

        kp = orb.detect(img, None)
        kp, orb_descriptors = orb.compute(img, kp)

        kernel_descriptors = []
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(img, kernel, mode='wrap')
            kernel_descriptors.extend((filtered[gt > 0].mean(), filtered[gt > 0].var()))

        haralick_feats = mahotas.features.haralick(img, distance=2)
        haralick_feats = list(haralick_feats.flatten())

        props = regionprops(gt.astype(np.uint8), cache=True)[0]
        props_list = [props.equivalent_diameter, props.eccentricity, props.extent]
        props_list += list(props.moments_central.flatten())

        gland_feats = list(orb_descriptors) + kernel_descriptors + haralick_feats + props_list
        features.append(gland_feats)

    return np.array(features)


def make_thumbnail(img, th_size, label=None):
    """
    :param img:
    :param th_size:
    :param label:
    :return: thumbnail

    Generate thumbnail of required size given an image, taking into account different gland sizes
    If label is given, colour the background with the label of specific classes
    """

    img = (img + 1) / 2 * 255  # change back to RGB from [-1, 1] range
    assert img.min() > 0.0 - 0.1 and img.max() < 255.0 + 0.1
    img = img.astype(np.uint8)
    img_mode = [int(mode(img[..., 0], axis=None)[0]),
                int(mode(img[..., 1], axis=None)[0]), int(mode(img[..., 2], axis=None)[0])]
    background = int(label) or 0
    if background == 2:
        img[img[..., 0] == img_mode[0], 0] = 100
        img[img[..., 1] == img_mode[1], 1] = 170
        img[img[..., 2] == img_mode[2], 0] = 100
    elif background == 3:
        img[img[..., 0] == img_mode[0], 0] = 100
        img[img[..., 1] == img_mode[1], 1] = 100
        img[img[..., 2] == img_mode[2], 2] = 170
    else:
        pass
    seg = np.logical_not(np.isclose(background, img)).any(axis=2).astype(np.uint8)  # isclose to compare floats
    gt2, contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    gland_cnt = contours[areas.index(max(areas))]
    x, y, w, h = cv2.boundingRect(gland_cnt)
    s = max(w, h)
    s = min(int(round(s * 1.5)), img.shape[0])  # increase bounding box for better visualization
    x = min(x, img.shape[0] - s)  # img is a square
    y = min(y, img.shape[0] - s)
    tn = img[y: y + s, x: x + s]
    # tn = np.array(Image.fromarray(tn).thumbnail(th_size, Image.ANTIALIAS)) # could upsample or downsample depending on original gland size
    tn = cv2.resize(tn, th_size, cv2.INTER_AREA)
    return tn.astype(np.uint8)


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots, 3))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def write_metadata(filename, labels):
    """
            Create a metadata file image consisting of sample indices and labels
            :param filename: name of the file to save on disk
            :param shape: tensor of labels
    """
    with open(filename, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("{}\t{}\n".format(index, label))

    print('Metadata file saved in {}'.format(filename))


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

ia.seed(7)
alpha = 0.3
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.8),  # horizontally flip
        iaa.Flipud(0.8),  # vertically flip
        # crop images by -5% to 10% of their height/width
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.05, 0.1),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-2, 2),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ),
        # # execute 0 to 5 of the following (less important) augmenters per image
        # # don't execute all of them, as that would often be way too strong
        # iaa.SomeOf((0, 5),
        #            [  # convert images into their superpixel representation
        #                # iaa.WithChannels([0, 1, 2],
        #                #                  iaa.OneOf([
        #                #                      iaa.GaussianBlur((0, 3.0)),
        #                #                      # blur images with a sigma between 0 and 3.0
        #                #                      iaa.AverageBlur(k=(2, 7)),
        #                #                      # blur image using local means with kernel sizes between 2 and 7
        #                #                      iaa.MedianBlur(k=(3, 11)),
        #                #                      # blur image using local medians with kernel sizes between 2 and 7
        #                #                  ])),
        #                iaa.WithChannels([0, 1, 2], iaa.Sharpen(alpha=(0, alpha), lightness=(0.75, 1.5))),
        #                # sharpen images
        #                iaa.WithChannels([0, 1, 2], iaa.Emboss(alpha=(0, alpha), strength=(0, 0.1))),  # emboss images
        #                # search either for all edges or for directed edges,
        #                # blend the result with the original image using a blobby mask
        #                iaa.WithChannels([0, 1, 2], iaa.SimplexNoiseAlpha(iaa.OneOf([
        #                    iaa.EdgeDetect(alpha=(0.05, alpha)),
        #                    iaa.DirectedEdgeDetect(alpha=(0.05, alpha), direction=(0.0, 1.0)),
        #                ]))),
        #                iaa.WithChannels([0, 1, 2],
        #                                 iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
        #                # add gaussian noise to images
        #                # iaa.WithChannels([0, 1, 2], #iaa.OneOf([
        #                # iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
        #                # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)])),
        #                iaa.WithChannels([0, 1, 2], iaa.Invert(0.05, per_channel=True)),  # invert color channels
        #                iaa.WithChannels([0, 1, 2], iaa.Add((-10, 10), per_channel=0.5)),
        #                # change brightness of images (by -10 to 10 of original value)
        #                iaa.WithChannels([0, 1, 2], iaa.AddToHueAndSaturation((-1, 1))),  # change hue and saturation
        #                # either change the brightness of the whole image (sometimes
        #                # per channel) or change the brightness of subareas
        #                iaa.WithChannels([0, 1, 2], iaa.OneOf([
        #                    iaa.Multiply((0.8, 1.3), per_channel=0.5),
        #                    iaa.FrequencyNoiseAlpha(
        #                        exponent=(-2, 2),
        #                        first=iaa.Multiply((0.8, 1.3), per_channel=True),
        #                        second=iaa.ContrastNormalization((0.5, 2.0)))
        #                ])),
        #                iaa.WithChannels([0, 1, 2], iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
        #                # improve or worsen the contrast
        #                iaa.WithChannels([0, 1, 2], iaa.Grayscale(alpha=(0.0, alpha))),
        #                # move pixels locally around (with random strengths)
        #                # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
        #                # sometimes move parts of the image around
        #                # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        #            ],
        #            random_order=True
        #            )
    ],
    random_order=True
)
seq = seq.to_deterministic()  # to ensure that every image is augmented in the same way (so that features are more comparable)
def augment_glands(imgs, gts, N=1):
    """
    :param imgs:
    :param gts:
    :param N:
    :return: Original images + N augmented copies of images and ground truths
    """
    if N:
        M = imgs.shape[0]
        # Does nothing for N = 0
        imgs = (imgs + 1) / 2 * 255
        order = [m + M*n for m in range(M) for n in range(N+1)] # reorder images so that images for same gland are consecutive
        cat = np.concatenate([imgs, gts], axis=3)
        cat = np.tile(cat, (N,1,1,1))
        out = seq.augment_images(cat)  # works on batch of images
        gts = np.concatenate((gts, out[..., 3:4]), axis=0)  # need to keep channel dim
        imgs = np.concatenate((imgs, out[..., 0:3]), axis=0)
        imgs, gts = imgs[order], gts[order]
        imgs = (imgs / 255 - 0.5)/0.5  # from -1 to 1
    return imgs, gts


def to_tensor(na):
    r"""Convert ndarray in sample to Tensors."""
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    if len(na.shape) == 2:
        na = na[..., np.newaxis]
    na = na.transpose((0, 3, 1, 2))  # grayscale or RGB
    na = torch.from_numpy(na.copy()).type(torch.FloatTensor)
    return na


def main(FLAGS):
    # Set segmentation network up
    if cuda.is_available():
        if isinstance(FLAGS.gpu_ids, Integral):
            cuda.set_device(FLAGS.gpu_ids)
        else:
            cuda.set_device(FLAGS.gpu_ids[0])

    ckpt_path = Path(FLAGS.seg_model).parent  # scope? does this change inside train and validate functs as well?
    try:
        LOADEDFLAGS = get_flags(str(ckpt_path / ckpt_path.parent.name) + ".txt")
        FLAGS.network_id = LOADEDFLAGS.network_id
        FLAGS.num_filters = LOADEDFLAGS.num_filters
        FLAGS.num_class = LOADEDFLAGS.num_class
    except FileNotFoundError as err:
        print("Settings could not be loaded - using network {} with {} filters".format(FLAGS.network_id,
                                                                                       FLAGS.num_filters))

    parallel = not isinstance(FLAGS.gpu_ids, Integral) and len(FLAGS.gpu_ids) > 1 and cuda.is_available()
    inputs = {'num_classes': FLAGS.num_class, 'num_channels': FLAGS.num_filters}
    if FLAGS.network_id == "UNet1":
        net = UNet1(**inputs).cuda() if cuda.is_available() else UNet1(**inputs)
    elif FLAGS.network_id == "UNet2":
        net = UNet2(**inputs).cuda() if cuda.is_available() else UNet2(**inputs)
    elif FLAGS.network_id == "UNet3":
        net = UNet3(**inputs).cuda() if cuda.is_available() else UNet3(**inputs)
    elif FLAGS.network_id == "UNet4":
        net = UNet4(**inputs).cuda() if cuda.is_available() else UNet4(**inputs)

    dev = None if cuda.is_available() else 'cpu'
    state_dict = load(os.path.expanduser(FLAGS.seg_model), map_location=dev)
    state_dict = {key[7:]: value for key, value in
                  state_dict.items()}  # never loading parallel as feature_net is later loaded as parallel
    net.load_state_dict(state_dict)
    net.eval()

    feature_blocks = ['input_block', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'enc6', 'center']
    features_net = FeatureExtractor(net, feature_blocks)
    if cuda.is_available():
        features_net = features_net.cuda()
    if parallel:
        features_net = DataParallel(features_net, device_ids=FLAGS.gpu_ids).cuda()
    features_net.eval()  # extracting features only

    # Setting ck5 network up:
    net_ck5 = Inception(num_classes=3)
    net_ck5.load_state_dict(os.path.expanduser(FLAGS.ck_model))
    net_ck5.eval()
    net_ck5 = FeatureExtractor(net_ck5, layers="fill")
    net_ck5.eval()

    datasets = [GlandPatchDataset(FLAGS.data_dir, mode, return_cls=True) for mode in ['train', 'val', 'test']]
    loaders = [DataLoader(ds, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.workers) for ds in datasets]

    tqdm.write("Generating features and thumbnails ...")
    X = []  # feature matrix
    X_alg = []
    thumbnails = []
    th_size = (FLAGS.thumbnail_size,) * 2
    labels = []
    
    N = FLAGS.augment_num
    for loader, mode in zip(loaders, ['train', 'val', 'test']):
        tqdm.write("... for {} data ...".format(mode))
        # Use train test and validate datasets
        for i, data in enumerate(tqdm(loader)):

            inputs = data[0]
            gts = data[1]
            colours_n_size = data[2].numpy()
            label_batch = data[3]
            label_batch = list(label_batch.numpy())
            M = inputs.shape[0]  # remember how many different gland images are there

            assert np.any(np.array(inputs.shape))  # non empty data

            imgs = inputs.numpy().transpose(0, 2, 3, 1)
            gts = gts.numpy().transpose(0, 2, 3, 1)
            imgs, gts = augment_glands(imgs, gts, N=N)  # consecutive images are augmented versions of first image

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(M,N+1)
            # for m in range(M):
            #     for n in range(N+1):
            #         #print(m * (N + 1) + n)
            #         axes[m,n].imshow(imgs[m * (N + 1) + n, ...])

            with torch.no_grad():  # speeds up computations and uses less RAM
                inputs = to_tensor(imgs)
                inputs = Variable(inputs)
                inputs = inputs.cuda() if cuda.is_available() else inputs
                feature_maps = features_net(inputs)  # Channel as last dimension

                assert feature_maps  # non empty feature map list
                if FLAGS.torch_feats:
                    gts = to_tensor(gts)
                    gts = gts.cuda() if cuda.is_available() else gts
                    glands_features = summary_net_features_torch(gts, feature_maps)
                else:
                    feature_maps = [block_fms.cpu().numpy() for block_fms in feature_maps]
                    glands_features = summary_net_features(gts, feature_maps)

                alg_feats = feature_extraction(imgs, gts)

                if N:
                    # started with M images, added M*N with augmentation, now take mean of N+1 responses
                    glands_features = glands_features.reshape((glands_features.shape[0] // (N + 1),
                                                               (N + 1), -1))
                    if FLAGS.torch_feats:
                        glands_features = glands_features.mean(dim=1)
                    else:
                        glands_features = glands_features.mean(axis=1)

                    alg_feats = alg_feats.reshape((glands_features.shape[0] // (N + 1),
                                                               (N + 1), -1))
                    alg_feats = alg_feats.mean(axis = 1)

            assert glands_features.shape[0] == M  # ensure there is one feature vector per image

            bad_imgs = []
            for j, gl_features, gl_alg_features in enumerate(zip(glands_features, alg_feats)):
                # Checks FIXME should not have to do this
                if FLAGS.torch_feats:
                    gl_features = gl_features.cpu().numpy()
                if gl_features.dtype != np.float32 or gl_features.ndim != 1:
                    bad_imgs.append(j)  # nan response
                    continue
                X.append(gl_features)  # feature vector
                X_alg.append(gl_alg_features)
                X[-1] = np.concatenate((gl_features, colours_n_size[j, ...]))

            # Save original images
            for j, (img, label) in enumerate(zip(imgs[0::(N+1), ...], label_batch)):  # only original images
                if j in bad_imgs:
                    del label_batch[j]
                    continue  # skip if nan response
                tn = make_thumbnail(img, th_size, label=label)
                thumbnails.append(tn)

                if i == 0:
                    imwrite(Path(FLAGS.save_dir).expanduser() / "thumbnail_test.png", tn)

            labels.extend(label_batch)

            assert len(X) == len(thumbnails)
            assert len(X) == len(labels)

    # Save features
    X = np.array(X)
    header = "mean_1,std_1,max_1,mean_2,std_2,max_2,..._{}x{}_{}".format(X.shape[0], X.shape[1], FLAGS.snapshot)
    save_file = Path(FLAGS.save_dir).expanduser() / ("feats_" + FLAGS.snapshot[:-4] + ".csv")
    with open(str(save_file), 'w') as feature_file:
        np.savetxt(feature_file, X, delimiter=' ', header=header)
    print("Saved feature matrix ({}x{})".format(*X.shape))

    # Save manual features
    X = np.array(X)
    header = "mean_1,std_1,max_1,mean_2,std_2,max_2,..._{}x{}_{}".format(X.shape[0], X.shape[1], FLAGS.snapshot)
    save_file = Path(FLAGS.save_dir).expanduser() / ("man_feats_" + FLAGS.snapshot[:-4] + ".csv")
    with open(str(save_file), 'w') as man_feature_file:
        np.savetxt(man_feature_file, X_alg, delimiter=' ', header=header)
    print("Saved manual feature matrix ({}x{})".format(*X.shape))

    # Save thumbnails
    thumbnails = np.array(thumbnails)
    spriteimage = create_sprite_image(thumbnails).astype(np.uint8)
    save_file = Path(FLAGS.save_dir).expanduser() / ("sprite_" + FLAGS.snapshot[:-4] + ".png")
    imwrite(save_file, spriteimage)
    save_file = Path(FLAGS.save_dir).expanduser() / (
                "thumbnails_" + FLAGS.snapshot[:-4] + ".npy")  # format for saving numpy data (not compressed)
    # Revert to RGB:
    thumbnails = thumbnails.astype(np.uint8)
    with open(str(save_file), 'wb') as thumbnails_file:  # it uses pickle, hence need to open file in binary mode
        np.save(thumbnails_file, thumbnails)
    print("Saved {} ({}x{} thumbnails".format(*thumbnails.shape))

    labels = np.array(labels)
    # Save labels:
    save_file = Path(FLAGS.save_dir).expanduser() / ("labels_" + FLAGS.snapshot[:-4] + ".tsv")
    with open(str(save_file), 'w') as labels_file:
        np.savetxt(labels_file, labels, delimiter='\t')
    print(
        "Saved labels ({}) - {} tumour and {} gland".format(labels.size, np.sum(labels == 2), np.sum(labels == 3)))
    print("Done!")


def thumbnails_size_check(str_size):
    size = int(str_size)
    if size > 100:
        warn("Max allowed thumbnail size is 100")
        size = 100
    return size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-segm', '--seg_model', required=True, type=str, help='segmentation model file')
    parser.add_argument('-ckm', '--ck_model', required=True, type=str, help='segmentation model file')
    parser.add_argument('-sd', '--save_dir', default="/gpfs0/well/win/users/achatrian/ProstateCancer/Results")
    parser.add_argument('--thumbnail_size', default=64, type=thumbnails_size_check)
    parser.add_argument('--augment_num', type=int, default=0)
    parser.add_argument('--torch_feats', type=str2bool, default='y')

    parser.add_argument('--network_id', type=str, default="UNet4")
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/ProstateCancer/Dataset")
    parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')
    parser.add_argument('--workers', default=2, type=int, help='the number of workers to load the data')

    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('-nf', '--num_filters', type=int, default=64, help='mcd number of filters for unet conv layers')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)