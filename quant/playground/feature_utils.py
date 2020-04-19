import sys

import numpy as np
from scipy.stats import mode
import torch
from torch import cuda
from torch import nn
import cv2
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from imgaug import augmenters as iaa
import imgaug as ia
from torch.nn.functional import interpolate
from torch.nn import functional as F
import mahotas

sys.path.append("../ck5")

from skimage.measure import regionprops

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractorUNet, self).__init__()
        self.submodule = submodule

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]

class FeatureExtractorUNet(nn.Module):
    """
    Class for extracting features from desired submodules in neural network.
    """

    def __init__(self, submodule, extracted_layers, layers=None):
        super(FeatureExtractorUNet, self).__init__()
        self.submodule = submodule
        if extracted_layers == "fill":
            self.extracted_layers = list(self.submodule._modules.keys())
        else:
            self.extracted_layers = extracted_layers.copy()

    def forward(self, x):
        outputs = []

        input_block = self.submodule.input_block(x)
        enc1 = self.submodule.enc1(input_block)
        enc2 = self.submodule.enc2(enc1)
        enc3 = self.submodule.enc3(enc2)
        enc4 = self.submodule.enc4(enc3)
        enc5 = self.submodule.enc5(enc4)
        enc6 = self.submodule.enc6(enc5)
        center = self.submodule.center(enc6)
        dec6 = self.submodule.dec6(torch.cat([center, F.upsample(enc6, center.size()[2:], mode='bilinear')], 1))
        dec5 = self.submodule.dec5(torch.cat([dec6, F.upsample(enc5, dec6.size()[2:], mode='bilinear')], 1))
        dec4 = self.submodule.dec4(torch.cat([dec5, F.upsample(enc4, dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.submodule.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.submodule.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.submodule.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final0 = self.submodule.final0(F.upsample(dec1, x.size()[2:], mode='bilinear'))
        final1 = self.submodule.final1(F.upsample(final0, x.size()[2:], mode='bilinear'))
        for layer in self.extracted_layers:
            outputs.append(eval(layer).detach())
        return outputs


class FeatureExtractorInception(nn.Module):
    """
    Class for extracting features from desired submodules in neural network.
    """

    def __init__(self, submodule):
        super(FeatureExtractorInception, self).__init__()
        self.submodule = submodule

    def forward(self, x):
        outputs = []
        if self.submodule.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.submodule.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.submodule.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.submodule.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.submodule.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.submodule.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.submodule.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.submodule.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.submodule.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.submodule.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.submodule.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.submodule.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.submodule.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.submodule.Mixed_6e(x)
        outputs.append(x)
        # 17 x 17 x 768
        if self.submodule.training and self.submodule.aux_logits:
            aux = self.submodule.AuxLogits(x)
        # 17 x 17 x 768
        x = self.submodule.Mixed_7a(x)
        outputs.append(x)
        # 8 x 8 x 1280
        x = self.submodule.Mixed_7b(x)
        outputs.append(x)
        # 8 x 8 x 2048
        x = self.submodule.Mixed_7c(x)
        outputs.append(x)

        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        outputs.append(x)  # this needs special section in feature extraction code
        x = self.submodule.fc(x)
        # 1000 (num_classes)
        if self.training and self.submodule.aux_logits:
            return x, aux
        return outputs


class FeatureExtractSummarise(nn.Module):

    def __init__(self,net, extracted_layers, N):
        super(FeatureExtractSummarise, self).__init__()
        self.fe = FeatureExtractorUNet(net, extracted_layers)
        self.summ = SummariserOfFeatures(N)

    def forward(self, x, gts):
        out = self.fe(x)
        return self.summ(out, gts)


def get_gland_bb(gt):
    """
    :param gt:
    :return: bb for largest area in images, assumed to be gland in dataset
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
            fm_gland = torch.zeros(fm.shape[0], 1)
            if cuda.is_available():
                fm_gland = fm_gland.cuda()
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


class SummariserOfFeatures(nn.Module):
    def __init__(self, N):
        super(SummariserOfFeatures, self).__init__()
        self.N = N

    def forward(self, feature_maps, gts):
        glands_features = self.summary_net_features_torch(gts, feature_maps)
        if self.N:
            # started with M images, added M*N with augmentation, now take mean of N+1 responses
            glands_features = glands_features.reshape((glands_features.shape[0] // (self.N + 1),
                                                       (self.N + 1), -1))
            glands_features = glands_features.mean(dim=1)
        return glands_features

    @staticmethod
    def summary_net_features_torch(gts, feature_maps):
        """
        :param gts: NCHW
        :param feature_maps: list of tensors [NCHW, NCHW, ...]
        :downsamplings: list of downsizes to calculates gts for
        :return: gland feature vectors from path feature maps

        Features are mean, median, var
        """

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
# orb = cv2.ORB_create()  # ORB features

kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 5):
        for frequency in (0.05, 0.25):
            # Gabor patches
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernel = np.tile(kernel[...,np.newaxis], (1,1,3))
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
        gt = gt.squeeze()
        if not gt.any():
            gt = np.ones(gt.shape, dtype=np.bool)

        # kp = orb.detect(img, None)
        # kp, orb_descriptors = orb.compute(img, kp)

        kernel_descriptors = []
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(img, kernel, mode='wrap')
            kernel_descriptors.extend((filtered[gt > 0].mean(), filtered[gt > 0].var()))

        haralick_feats = mahotas.features.haralick(img.astype(np.uint8), distance=2)
        haralick_feats = list(haralick_feats.flatten())

        props = regionprops(gt.astype(np.uint8), cache=True)[0]
        props_list = [props.equivalent_diameter, props.eccentricity, props.extent]
        props_list += list(props.moments_central.flatten())

        gland_feats = kernel_descriptors + haralick_feats + props_list
        features.append(gland_feats)
    assert features
    return np.array(features)


def make_thumbnail(img, gt, th_size, label=None):
    """
    :param img:
    :param th_size:
    :param label:
    :return: thumbnail

    Generate thumbnail of required size given an images, taking into account different gland sizes
    If label is given, colour the background with the label of specific classes
    """

    img = (img + 1) / 2 * 255  # change back to RGB from [-1, 1] range
    assert img.min() > 0.0 - 0.1 and img.max() < 255.0 + 0.1
    img = img.astype(np.uint8)
    img_mode = [int(mode(img[np.logical_not(gt), 0], axis=None)[0]),
                int(mode(img[np.logical_not(gt), 1], axis=None)[0]),
                int(mode(img[np.logical_not(gt), 2], axis=None)[0])]
    background = int(label) or 0
    if label == 2:
        img[img[..., 0] == img_mode[0], 0] = 100
        img[img[..., 1] == img_mode[1], 1] = 170
        img[img[..., 2] == img_mode[2], 0] = 100
    elif label == 3:
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


def make_thumbnail_fullcut(img, gt, th_size, label=None):
    """
    :param img:
    :param th_size:
    :param label:
    :return: thumbnail

    Generate thumbnail of required size given an images, taking into account different gland sizes
    If label is given, colour the background with the label of specific classes
    """

    img = (img + 1) / 2 * 255  # change back to RGB from [-1, 1] range
    assert img.min() > 0.0 - 0.1 and img.max() < 255.0 + 0.1
    img = img.astype(np.uint8)
    background = int(label) or 0
    kernel = np.ones((8, 8), np.uint8)
    gt2 = cv2.erode(np.uint8(gt > 0), kernel, iterations=1)
    gt2, contours, hierarchy = cv2.findContours(gt2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cX, cY = 0, 0
    center = (img.shape[0] / 2, img.shape[1] / 2)
    best_dist = abs(cX - center[1]) + abs(cY - center[0])

    # Find central gland
    for contour in contours:
        M = cv2.moments(contour)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            pass

        dist_new = abs(cX - center[1]) + abs(cY - center[0])
        if dist_new < best_dist:
            best_dist = dist_new
            gland_contour = contour

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [gland_contour], -1, 1, -1)
    mask = cv2.dilate(mask, kernel, iterations=2)  # dilate to increase again (2 iters)
    img = cv2.bitwise_and(img, img, mask=mask)

    # Colour background with label
    mask = np.logical_not(mask)
    if background == 2:
        img[mask, 0] = 100
        img[mask, 1] = 170
        img[mask, 0] = 100
    elif background == 1:
        img[mask, 0] = 100
        img[mask, 1] = 100
        img[mask, 2] = 170
    else:
        pass

    # Make thumbnail
    x, y, w, h = cv2.boundingRect(gland_contour)
    s = max(w, h)
    s = min(int(round(s * 1.3)), img.shape[0])  # increase bounding box for better visualization
    x = min(x, img.shape[0] - s)  # img is a square
    y = min(y, img.shape[0] - s)
    tn = img[y: y + s, x: x + s]
    tn = cv2.resize(tn, th_size, cv2.INTER_AREA)
    return tn.astype(np.uint8)


def create_sprite_image(images):
    """Returns a sprite images consisting of images passed as argument. Images should be count x width x height"""
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
            Create a metadata file images consisting of sample indices and labels
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
            mode=ia.ALL  # use any of scikit-images's warping modes (see 2nd images from the top for examples)
        ),
    ],
    random_order=True
)
seq = seq.to_deterministic()  # to ensure that every images is augmented in the same way (so that features are more comparable)
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
        if gts.ndim < imgs.ndim:
            gts = gts[..., np.newaxis]
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
    # numpy images: H x W x C
    # torch images: C X H X W
    if len(na.shape) == 2:
        na = na[..., np.newaxis]
    na = na.transpose((0, 3, 1, 2))  # grayscale or RGB
    na = torch.from_numpy(na.copy()).type(torch.FloatTensor)
    return na