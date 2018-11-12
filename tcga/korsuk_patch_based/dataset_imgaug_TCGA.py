import glob
import os
import cv2
import random
import numbers
import torch

import numpy as np
from torch.utils.data import Dataset

import imgaug as ia
from imgaug import augmenters as iaa

class ImgDataset(Dataset):

    def __init__(self, dir_, mode, cv):
        self.mode = mode

        self.file_list = []
        self.label = []

        file_list = []
        label_list = []
        for file, label in zip(cv[mode]['img'], cv[mode]['label']):
            temp = glob.glob(os.path.join(dir_, file, '*.png'))

            if label == 'CMS1':
                label = 0
            elif label == 'CMS2':
                label = 1
            elif label == 'CMS3':
                label = 2
            elif label == 'CMS4':
                label = 3
            else:
                continue

            file_list += temp
            label_list += [label] * len(temp)

        self.file_list = file_list
        self.label = label_list

        self.dir = dir_
        self.scale = Scale(299)

        ##############################
        # weighting
        nclasses = len(np.unique(self.label))
        count = [0] * nclasses
        for item in self.label:
            count[item] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(self.label)
        for idx, val in enumerate(self.label):
            weight[idx] = weight_per_class[val]

        self.weight = weight
        ##############################

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [  # convert images into their superpixel representation
                               iaa.WithChannels([0, 1, 2],
                                                iaa.OneOf([
                                                    iaa.GaussianBlur((0, 3.0)),
                                                    # blur images with a sigma between 0 and 3.0
                                                    iaa.AverageBlur(k=(2, 7)),
                                                    # blur image using local means with kernel sizes between 2 and 7
                                                    iaa.MedianBlur(k=(3, 11)),
                                                    # blur image using local medians with kernel sizes between 2 and 7
                                                ])),
                               iaa.WithChannels([0, 1, 2], iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                               # sharpen images
                               iaa.WithChannels([0, 1, 2], iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),
                               # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.WithChannels([0, 1, 2], iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ]))),
                               iaa.WithChannels([0, 1, 2], iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                                                     per_channel=0.5)),
                               # add gaussian noise to images
                               iaa.WithChannels([0, 1, 2], iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)])),
                               iaa.WithChannels([0, 1, 2], iaa.Invert(0.05, per_channel=True)),  # invert color channels
                               iaa.WithChannels([0, 1, 2], iaa.Add((-10, 10), per_channel=0.5)),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.WithChannels([0, 1, 2], iaa.AddToHueAndSaturation((-20, 20))),
                               # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.WithChannels([0, 1, 2], iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                   iaa.FrequencyNoiseAlpha(
                                       exponent=(-4, 0),
                                       first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                       second=iaa.ContrastNormalization((0.5, 2.0))
                                   )])),
                               iaa.WithChannels([0, 1, 2], iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
                               # improve or worsen the contrast
                               iaa.WithChannels([0, 1, 2], iaa.Grayscale(alpha=(0.0, 1.0))),
                               # move pixels locally around (with random strengths)
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # sometimes move parts of the image around
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        gt = self.label[idx]

        bgr_img = cv2.imread(img_name, -1)
        b, g, r = cv2.split(bgr_img)  # get b,g,r

        image = cv2.merge([r, g, b])  # switch it to rgb

        image = self.scale(image)
        # im aug
        image = self.seq.augment_image(image)
        if image.ndim < 3:
            image = cv2.merge([image, image, image])

        # scale between 0 and 1 and swap the dimension
        image = image.transpose(2, 0, 1) / 255.0

        # normalised image between -1 and 1
        # image = [np.expand_dims((img - 0.5) / 0.5, axis=0) for img in image]
        # image = np.concatenate(image, axis=0)

        # convert to torch tensor
        dtype = torch.FloatTensor
        image = torch.from_numpy(image).type(dtype)

        return image, gt


class Scale(object):
    """Rescales the input np.ndarray to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv.INTER_CUBIC
    """

    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(float(self.size) * h / w)
        else:
            oh = self.size
            ow = int(float(self.size) * w / h)
        return cv2.resize(img, dsize=(ow, oh),
                          interpolation=self.interpolation)


class CenterCrop(object):
    """Crops the given np.ndarray at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        return img[y1:y1 + th, x1:x1 + tw, :]


class RandomScale(object):

    def __init__(self, interpolation=cv2.INTER_CUBIC):
        self.interpolation = interpolation

    def __call__(self, img):
        # random_scale = random.sample([0.25, 0.5, 1.0], 1)
        random_scale = [1.0]
        w, h = img.shape[1], img.shape[0]
        w = int(w * random_scale[0])
        h = int(h * random_scale[0])

        return cv2.resize(img, dsize=(w, h),
                          interpolation=self.interpolation)


class RandomCrop(object):
    """Crops the given np.ndarray at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if img.ndim == 3:
            img = img[y1:y1 + th, x1:x1 + tw, :]
        else:
            img = img[y1:y1 + th, x1:x1 + tw]
        return img


class RandomCropCell(object):
    """Crops the given np.ndarray at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        if w == tw and h == th:
            return img

        y, x = np.where(img[:, :, 3] > 0)
        r = random.randint(0, len(x))

        x1 = x[r - 1] - 128
        y1 = y[r - 1] - 128

        if x1 >= w - tw or y1 >= h - th or x1 < 0 or y1 < 0:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

        if img.ndim == 3:
            img = img[y1:y1 + th, x1:x1 + tw, :]
        else:
            img = img[y1:y1 + th, x1:x1 + tw]
        return img


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given np.ndarray with a probability of 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 1).reshape(img.shape)
        return img


class RandomVerticalFlip(object):
    """Randomly vertically flips the given np.ndarray with a probability of 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 0).reshape(img.shape)
        return img


class RandomTransposeFlip(object):
    """Randomly horizontally and vertically flips the given np.ndarray with a probability of 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, -1).reshape(img.shape)
        return img


class RandomBlur(object):
    def __call__(self, img):
        label = img
        if random.random() < 0.8:
            # kernel_size = random.randrange(1, 19 + 1, 2)
            kernel_size = 19
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        return {'image': img, 'label': label}


class Convert(object):
    """Randomly horizontally and vertically flips the given np.ndarray with a probability of 0.5
    """

    def __call__(self, img):
        if img.ndim < 3:
            img = np.expand_dims(img, axis=2)
        img = img.transpose(2, 0, 1)

        dtype = torch.FloatTensor
        img = torch.from_numpy(img).type(dtype) / 255.0

        return img
