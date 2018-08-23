import glob
import os
import re
import cv2
import random
import numbers
from collections import namedtuple

import imageio
import torch
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from torchvision.transforms import Compose
from torch.utils.data import Dataset
import warnings


ia.seed(1)
Example = namedtuple('example', ['img', 'gt'])

###Pytorch dataset###
class ProstateDataset(Dataset):

    def __init__(self, dir, mode, out_size=1024, down=2.0, num_class=1, grayscale=False, augment=None, load_wm=False):
        self.mode = mode
        self.dir = dir
        self.out_size = out_size
        self.down = down
        self.num_class = num_class
        self.grayscale = grayscale
        self.augment = augment #set augment, or augment wehn training
        assert(mode in ['train', 'validate', 'test'])

        #Read data paths
        self.gt_files = glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9],[0-9].png'), recursive=True) #for 1,1 patch
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9][0-9],[0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9][0-9][0-9],[0-9][0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9][0-9][0-9][0-9],[0-9][0-9][0-9][0-9].png'), recursive=True))
        self.img_files = [re.sub('mask', 'img', filename) for filename in self.gt_files]
        self.weightmap_f = glob.glob('**/_weightmap_[0-9],[0-9].png', recursive=True)
        self.weightmap_f.extend(glob.glob('**/_weightmap_[0-9][0-9],[0-9][0-9].png', recursive=True))
        self.weightmap_f.extend(glob.glob('**/_weightmap_[0-9][0-9][0-9],[0-9][0-9][0-9].png', recursive=True))
        self.weightmap_f.extend(glob.glob('**/_weightmap_[0-9][0-9][0-9][0-9],[0-9][0-9][0-9][0-9].png', recursive=True))
        assert(self.gt_files); r"Cannot be empty"
        assert(self.img_files);

        #Augmentation sequence
        self.downsample = Downsample(down, min_size=out_size)
        if self.mode == 'train': self.crop_aug = RandomCrop(int(out_size*down))
        elif self.mode in ['validate', 'test']: self.center_crop = CenterCrop(int(out_size*down))

        if self.augment:
            geom_augs = [iaa.Affine(rotate=45),
                        iaa.Fliplr(0.8),
                        iaa.Flipud(0.8),
                        iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                        ]
            img_augs = [iaa.AdditiveGaussianNoise(scale=0.05*255),
                            iaa.Add(40, per_channel=True),
                            iaa.Dropout(p=(0, 0.2)),
                            iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
                            iaa.AverageBlur(k=(2, 7)),
                            iaa.MedianBlur(k=(3, 7)),
                            iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.75, 1.5)),
                            iaa.Emboss(alpha=(0.0, 0.5), strength=(0.5, 1.0)),
                            iaa.EdgeDetect(alpha=(0.0, 0.5))]
            if not self.grayscale:
                #Add color modifications
                img_augs.append(iaa.WithChannels(0, iaa.Add((10, 50))))
                img_augs.append(iaa.WithChannels(1, iaa.Add((10, 50))))
                img_augs.append(iaa.WithChannels(3, iaa.Add((10, 50))))
                img_augs.append(iaa.ContrastNormalization((0.5, 1.5)))
                img_augs.append(iaa.Multiply((0.5, 1.5)))
                img_augs.append(iaa.Invert(0.5))
                img_augs.append(iaa.Grayscale(alpha=(0.0, 1.0)))
            self.geom_aug_seq = iaa.SomeOf((0, None), geom_augs) #apply 0-all augmenters; both img and gt
            self.img_seq = iaa.SomeOf((0, None), img_augs) #only img

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = imageio.imread(self.img_files[idx])
        gt = imageio.imread(self.gt_files[idx])
        assert(len(set(gt.flatten())) <= 3); "Number of classes is greater than specified"

        #Pad images to desired size (for smaller tiles):
        size_b4_down = self.out_size * int(self.down)
        too_narrow = img.shape[1] < size_b4_down
        too_short = img.shape[0] < size_b4_down
        if too_narrow or too_short:
            delta_w = size_b4_down - img.shape[1] if too_narrow else 0
            delta_h = size_b4_down - img.shape[0] if too_short else 0
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            assert(img.shape[0] >= size_b4_down and img.shape[1] >= size_b4_down)  #Ensure padded
            assert(gt.shape[0] >= size_b4_down and gt.shape[1] >= size_b4_down)  #Ensure padded

        #Transform gt to desired number of classes
        if self.num_class > 1: gt = self.split_gt(gt)
        else: gt=(gt>0)[:,:,np.newaxis].astype(np.uint8)

        #Grascale
        if self.grayscale: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.mode == 'train':
            #Augment:
            img = self.crop_aug([img], random_state=1)[0] #crop to out_size * downsample
            img = self.downsample(img) #downsample to size out_sizee
            gt = self.crop_aug([gt], random_state=1)[0]
            gt = self.downsample(gt)
        else:
            img = self.center_crop(img) #center crop tiles when testing
            img = self.downsample(img) #downsample like in training
            gt = self.center_crop(gt)
            gt = self.downsample(gt)

        if self.augment:
            geom_seq_det = self.geom_aug_seq.to_deterministic() #ensure ground truth and image are transformed identically
            img = geom_seq_det.augment_image(img)
            img = self.img_seq.augment_image(img)
            img.clip(0, 255) #ensure added values don't stray from normal boundaries
            gt = geom_seq_det.augment_image(gt)

        return self.to_tensor(img, gt)

    def split_gt(self, gt, cls_values=[0,2,4], merge_cls={4:2}):
        cls_gts=[]

        if len(cls_values) == 2: #binarize
            gt = (gt > 0).astype(np.uint8)
            cls_values=[0,1]

        #Build one gt image per class
        for c in cls_values:
            map = np.array(gt == c, dtype=np.uint8)  #simple 0 or 1 for different classes
            #could weight pixels here
            cls_gts.append(map)

        #Handle overlapping classes (fill)
        for cs, ct in merge_cls.items():
            mask = cls_gts[cls_values.index(cs)] > 0
            cls_gts[cls_values.index(ct)][mask] = 1

        gt = np.stack(cls_gts, axis=2) #need to rescale with opencv later, so channel must be last dim
        return gt


    def to_tensor(self, img, gt):
        r"""Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img[np.newaxis,:,:] if self.grayscale else img.transpose((2, 0, 1)) #grayscale or RGB
        gt = gt.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).type(torch.FloatTensor)
        gt = torch.from_numpy(gt.copy()).type(torch.FloatTensor)
        return Example(img=img, gt=gt)


class RandomCrop(object):
    """Crops the given np.ndarray's at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, images, random_state, parents=None, hooks=None):
        cropped_images = []
        random.seed(random_state) #ensure that deterministic sequence applies the same transormation
        for img in images:
            w, h = img.shape[1], img.shape[0]
            th, tw = self.size
            if w <= tw: tw = w
            if h <= th: th = h
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            if img.ndim == 3:
                img = img[y1:y1+th, x1:x1+tw, :]
            else:
                img = img[y1:y1 + th, x1:x1 + tw]
            cropped_images.append(img)
        return cropped_images


#####
class Compose(object):
    "Apply set of transforms to image"
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

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

class Downsample(object):
    """Downsamples the input by a given factor
    interpolation: Default: cv.INTER_CUBIC
    """
    def __init__(self, factor, min_size=None, interpolation=cv2.INTER_CUBIC):
        self.factor = float(factor)
        self.min_size = min_size or 10000
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        try:
            c = img.shape[2]
        except IndexError:
            c = 0
        ow = int(np.ceil(w / self.factor))
        oh = int(np.ceil(w / self.factor))
        ow = ow if ow > self.min_size else self.min_size
        oh = oh if oh > self.min_size else self.min_size
        outimg = cv2.resize(img, dsize=(ow, oh),
                          interpolation=self.interpolation)
        if c == 1: outimg = outimg[:,:,np.newaxis]
        return outimg

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
        return img[y1:y1+th, x1:x1+tw, ...]


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


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = cv2.flip(img, 1).reshape(img.shape)
            mask = cv2.flip(img, 1).reshape(mask.shape)
        return img, mask

class RandomVerticalFlip(object):
    """Randomly vertically flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = cv2.flip(img, 0).reshape(img.shape)
            mask = cv2.flip(img, 0).reshape(mask.shape)
        return img, mask

class RandomTransposeFlip(object):
    """Randomly horizontally and vertically flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = cv2.flip(img, -1).reshape(img.shape)
            mask = cv2.flip(img, -1).reshape(mask.shape)
        return img, mask

class RandomBlur(object):
    def __call__(self, img, mask):
        if random.random() < 0.8:
            # kernel_size = random.randrange(1, 19 + 1, 2)
            kernel_size = 19
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        return img, mask


class Convert(object):
    """Randomly horizontally and vertically flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img, mask):
        if img.ndim < 3:
            img = np.expand_dims(img, axis=2)
        img = img.transpose(2, 0, 1)

        dtype = torch.FloatTensor
        img = torch.from_numpy(img).type(dtype)/255.0

        return img
