import torch.utils.data as data
import numbers
import random
import copy
import numpy as np
import cv2
import torch
from imgaug import augmenters as iaa


class BaseDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.paths = []  # store data in paths for make_subset to work, or give different store_name
        if self.opt.is_train:
            self.aug_seq = get_augment_seq(self.opt.augment_level)

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __len__(self):
        return len(self.paths)

    def get_sampler(self):
        r"""Abstract method, returns sampler for dataset, which is used in create_dataloader.
        If not overwritten it is ignored via the None flag"""
        return None

    def make_subset(self, selector='', selector_type='match', store_name='paths', deepcopy=False,
                    additional_stores=('labels',)):
        r"""
        :param selector: search for this pattern in data path (selector_type=='match), or pass the desired indices (selector_type=='indices')
        :param selector_type: 'match': searches for patterns in path strings; 'indices' selects the desired indices
        :param store_name: name of attribute containing paths to data
        :param deepcopy: whether to create a new dataset object
        :param additional_stores: additional stores to be subsampled
        :return:
        """
        dataset = copy.deepcopy(self) if deepcopy else self
        if selector_type == 'match':
            indices = [i for i, path in enumerate(getattr(dataset, store_name)) if selector in str(path)]  # NB only works for datasets that store paths in self.paths
        elif selector_type == 'indices':
            indices = selector
        else:
            raise NotImplementedError(f"Unknown selector type '{selector_type}'")
        lengths = set()
        for store_name_ in additional_stores + (store_name,):
            try:
                store = getattr(dataset, store_name_)
                lengths.add(len(store))
                assert len(lengths) == 1, "All stores must have equal length"
                if not indices:
                    raise ValueError("Cannot make subset from empty index set")
                if not store:
                    raise ValueError(f"{dataset.name()}().{store_name_} is empty")
                store = tuple(store[i] for i in indices)
                setattr(dataset, store_name_, store)
            except AttributeError:
                print(f"{store_name_} not defined in {dataset.name()}")
                raise
            except IndexError:
                print(f"Max index '{max(indices)}' greater than {store_name_} length ({len(store)}))")
                raise
            if len(store) == 0:
                raise ValueError("Subset is empty - could be due to train/test split")
        print(f"subset of len = {len(store)} was created")
        return dataset

    def setup(self):
        pass

    def augment_image(self, image, ground_truth=None):
        r"""
        Augment image and ground truth using desired augmentation sequence
        :param image:
        :param ground_truth:
        :return:
        """
        if self.opt.augment_level:
            seq_det = self.aug_seq.to_deterministic()  # needs to be called for every batch https://github.com/aleju/imgaug
            image = seq_det.augment_image(image)
            if ground_truth is not None:
                if ground_truth.ndim == 2:
                    ground_truth = np.tile(ground_truth[..., np.newaxis], (1, 1, 3))
                ground_truth = np.squeeze(seq_det.augment_image(ground_truth, ground_truth=True))
                ground_truth = ground_truth[..., 0]
        return image, ground_truth

# Transforms


class AugSeq:

    def __init__(self, seq_geom: iaa.Sequential, seq_content: iaa.Sequential):
        self.seq_geom = seq_geom
        self.seq_full = seq_content

    def augment_image(self, image, ground_truth=False):
        if ground_truth:
            return self.seq_geom.augment_image(image)
        else:
            return self.seq_geom.augment_image(self.seq_full.augment_image(image))

    def to_deterministic(self, geom=True, full=False):
        r"""Return an augmentation sequence that always performs the same geometric transforms"""
        return AugSeq(
            self.seq_geom.to_deterministic() if geom else self.seq_geom,
            self.seq_full if full else self.seq_full
        )


def get_augment_seq(augment_level):
    r"""
    Generates an imgaug augmentation sequence. The strength of the augment
    :param augment_level:
    :return:
    """

    if not 0 < augment_level < 5:
        raise ValueError("Level of augmentation must be between 1 and 4 (input was {})".format(augment_level))

    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    if augment_level == 1:
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=["reflect", "symmetric"],
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    mode=["reflect", "symmetric"]
                    # use any of scikit-images's warping modes (see 2nd images from the top for examples)
                ))]
        )
        aug_seq = AugSeq(seq, seq)
    elif augment_level == 2:
        # was missing WithChannels
        # no superpixels in level 2 nor hue inversion
        aug_seq = AugSeq(
            iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                    iaa.Flipud(0.2),  # vertically flip 20% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.05),
                        pad_mode=["reflect", "symmetric"],
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-5, 5),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        mode=["reflect", "symmetric"]
                        # use any of scikit-images's warping modes (see 2nd images from the top for examples)
                    ))
                ]),
            iaa.Sequential(
            [

             iaa.SomeOf((0, 2),
                        [
                            # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                            iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                            ]),
                            # convert images into their superpixel representation
                            iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur images using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur images using local medians with kernel sizes between 2 and 7
                            ]),
                            iaa.LinearContrast(alpha=(0.5, 1.0), per_channel=0.5),
                            # improve or worsen the contrast
                            iaa.Grayscale(alpha=(0.0, 1.0)),
                            sometimes(iaa.ElasticTransformation(alpha=(0.1, 0.3), sigma=0.2)),
                            # move pixels locally around (with random strengths)
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))),
                            # sometimes move parts of the images around
                            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.03)))
                       ]
                       )

            ]
        )
        )

    elif augment_level == 3:
        aug_seq = AugSeq(
            iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                    iaa.Flipud(0.2),  # vertically flip 20% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=["reflect", "symmetric"],
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        mode=["reflect", "symmetric"]
                        # use any of scikit-images's warping modes (see 2nd images from the top for examples)
                    ))
                ]),
            iaa.Sequential(
            [
             iaa.SomeOf((0, 5),
                        [
                            sometimes(iaa.Superpixels(p_replace=(0, 0.2), n_segments=(5, 50))),
                            # convert images into their superpixel representation
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 3.0)),
                                # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(2, 7)),
                                # blur images using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 11)),
                                # blur images using local medians with kernel sizes between 2 and 7
                            ]),
                            # THESE DON'T WORK IN SUBPROCESSES
                            # iaa.Sharpen(alpha=(0, alpha), lightness=(0.75, 1.5)),  # sharpen images
                            # iaa.Emboss(alpha=(0, alpha), strength=(0, 2.0)),  # emboss images
                            # # search either for all edges or for directed edges,
                            # # blend the result with the original images using a blobby mask
                            # iaa.SimplexNoiseAlpha(iaa.OneOf([
                            #     iaa.EdgeDetect(alpha=(0.2, alpha)),
                            #     iaa.DirectedEdgeDetect(alpha=(0.2, alpha), direction=(0.0, 1.0)),
                            # ])),
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                      per_channel=0.5),
                            # add gaussian noise to images
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                # randomly remove up to 10% of the pixels
                                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05),
                                                  per_channel=0.2),
                            ]),
                            iaa.Invert(0.1, per_channel=True),  # invert color channels
                            iaa.Add((-30, 30), per_channel=0.5),
                            # change brightness of images (by -10 to 10 of original value)
                            iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                            # either change the brightness of the whole images (sometimes
                            # per channel) or change the brightness of subareas
                            iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                            # improve or worsen the contrast
                            # iaa.Grayscale(alpha=(0.0, 1.0)),
                            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                            # move pixels locally around (with random strengths)
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                            # sometimes move parts of the images around
                            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                        ])
            ])
        )

    elif augment_level == 4:
        aug_seq = AugSeq(
            iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                    iaa.Flipud(0.2),  # vertically flip 20% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=["reflect", "symmetric"],
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        mode=["reflect", "symmetric"]
                        # use any of scikit-images's warping modes (see 2nd images from the top for examples)
                    ))
                ]),
            iaa.Sequential(
                [
                 iaa.SomeOf((0, 7),
                            [
                                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                                # convert images into their superpixel representation
                                iaa.OneOf([
                                    iaa.GaussianBlur((0, 3.0)),
                                    # blur images with a sigma between 0 and 3.0
                                    iaa.AverageBlur(k=(2, 7)),
                                    # blur images using local means with kernel sizes between 2 and 7
                                    iaa.MedianBlur(k=(3, 11)),
                                    # blur images using local medians with kernel sizes between 2 and 7
                                ]),
                                # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images  # { REMOVED AS NOT WORKING ON MULTIPROCESSING https://github.com/aleju/imgaug/issues/147
                                # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                                # search either for all edges or for directed edges,
                                # blend the result with the original images using a blobby mask
                                # iaa.SimplexNoiseAlpha(iaa.OneOf([
                                #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                                # ])),                                                                  # }
                                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                          per_channel=0.5),
                                # add gaussian noise to images
                                iaa.OneOf([
                                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                    # randomly remove up to 10% of the pixels
                                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05),
                                                      per_channel=0.2),
                                ]),
                                iaa.Invert(0.05, per_channel=True),  # invert color channels
                                iaa.Add((-10, 10), per_channel=0.5),
                                # change brightness of images (by -10 to 10 of original value)
                                iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                                # either change the brightness of the whole images (sometimes
                                # per channel) or change the brightness of subareas
                                iaa.OneOf([
                                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                    iaa.BlendAlphaFrequencyNoise(
                                        exponent=(-4, 0),
                                        foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                                        background=iaa.LinearContrast((0.5, 2.0))),
                                ]),
                                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                                # improve or worsen the contrast
                                # iaa.Grayscale(alpha=(0.0, 1.0)),
                                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                                # move pixels locally around (with random strengths)
                                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                                # sometimes move parts of the images around
                                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                            ],
                            random_order=True
                            )
                ],
                random_order=True
            )
        )
    return aug_seq


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
        self.last_crop = (0, 0)  # keep track of cropping offset, accessible from crop object

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
        self.last_crop = (x1, y1)
        return img


#####
class Compose(object):
    "Apply set of transforms to images"
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
    For example, if height > width, then images will be
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
        self.last_crop = (x1, y1)
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
