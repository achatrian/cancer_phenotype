import os.path
import random
import torchvision.transforms as transforms
import torch
from .base_dataset import BaseDataset
from .image_folder import make_dataset
from PIL import Image
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np



class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.cnt = 0
        self.split_gt = split_gt
        if opt.isTrain:
            self.seq0 = iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                    iaa.Flipud(0.4),  # vertically flip 20% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=["reflect", "symmetric"],
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
                        mode=["reflect", "symmetric"]
                        # use any of scikit-images's warping modes (see 2nd images from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per images
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((0, 3),
                               [  # convert images into their superpixel representation
                                   iaa.WithChannels([0, 1, 2],
                                                    iaa.OneOf([
                                                        iaa.GaussianBlur((0, 0.3)),
                                                        # blur images with a sigma between 0 and 3.0
                                                        iaa.AverageBlur(k=(2, 3)),
                                                        # blur images using local means with kernel sizes between 2 and 7
                                                        iaa.MedianBlur(k=(3, 3)),
                                                        # blur images using local medians with kernel sizes between 2 and 7
                                                    ])),
                                   iaa.WithChannels([0, 1, 2], iaa.Sharpen(alpha=(0, 0.3), lightness=(0.9, 1.1))),
                                   # sharpen images
                                   iaa.WithChannels([0, 1, 2], iaa.Emboss(alpha=(0, 0.3), strength=(0, 0.3))),
                                   # emboss images
                                   # search either for all edges or for directed edges,
                                   # blend the result with the original images using a blobby mask
                                   iaa.WithChannels([0, 1, 2], iaa.SimplexNoiseAlpha(iaa.OneOf([
                                       iaa.EdgeDetect(alpha=(0.05, 0.1)),
                                       iaa.DirectedEdgeDetect(alpha=(0.05, 0.1), direction=(0.0, 1.0)),
                                   ]))),
                                   iaa.WithChannels([0, 1, 2],
                                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                                              per_channel=0.5)),
                                   # add gaussian noise to images
                                   #iaa.WithChannels([0, 1, 2], iaa.OneOf([
                                       #iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                       #iaa.CoarseDropout((0.01, 0.05), size_percent=(0.02, 0.05), per_channel=0.2)])),
                                   #iaa.WithChannels([0, 1, 2], iaa.Invert(0.01, per_channel=True)),  # invert color channels
                                   iaa.WithChannels([0, 1, 2], iaa.Add((-4, 4), per_channel=0.5)),
                                   # change brightness of images (by -10 to 10 of original value)
                                   iaa.WithChannels([0, 1, 2], iaa.AddToHueAndSaturation((-4, 4))),
                                   # change hue and saturation
                                   # either change the brightness of the whole images (sometimes
                                   # per channel) or change the brightness of subareas
                                   # iaa.WithChannels([0, 1, 2], iaa.ContrastNormalization((0.1, 0.2), per_channel=0.5)),
                                   # improve or worsen the contrast
                                   iaa.WithChannels([0, 1, 2], iaa.Grayscale(alpha=(0.0, 0.1))),
                                   # move pixels locally around (with random strengths)
                                   sometimes(iaa.PiecewiseAffine(scale=(0.001, 0.005))),
                               ],
                               random_order=True
                               ),
                    sometimes(iaa.WithChannels([3, 4, 5], # augmentation on the GT
                                               iaa.OneOf([
                                                   iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                                   iaa.CoarseDropout((0.01, 0.05), size_percent=(0.02, 0.05), per_channel=0.2)])), ),
                    sometimes(iaa.WithChannels([3, 4, 5], iaa.PiecewiseAffine(scale=(0.001, 0.005))))
                ],
                random_order=True
            )

            self.seq1 = iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                    iaa.Flipud(0.4),  # vertically flip 20% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=["reflect", "symmetric", "edge"],
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
                        mode=["reflect", "symmetric", "edge"]  # use any of scikit-images's warping modes (see 2nd images from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per images
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((0, 5),
                               [  # convert images into their superpixel representation
                                   iaa.WithChannels([0, 1, 2],
                                                    iaa.OneOf([
                                                        iaa.GaussianBlur((0, 3.0)),
                                                        # blur images with a sigma between 0 and 3.0
                                                        iaa.AverageBlur(k=(2, 7)),
                                                        # blur images using local means with kernel sizes between 2 and 7
                                                        iaa.MedianBlur(k=(3, 11)),
                                                        # blur images using local medians with kernel sizes between 2 and 7
                                                    ])),
                                   iaa.WithChannels([0, 1, 2], iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                                   # sharpen images
                                   iaa.WithChannels([0, 1, 2], iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),
                                   # emboss images
                                   # search either for all edges or for directed edges,
                                   # blend the result with the original images using a blobby mask
                                   iaa.WithChannels([0, 1, 2], iaa.SimplexNoiseAlpha(iaa.OneOf([
                                       iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                       iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                                   ]))),
                                   iaa.WithChannels([0, 1, 2],
                                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
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
                                   # either change the brightness of the whole images (sometimes
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
                                   # sometimes move parts of the images around
                                   sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                               ],
                               random_order=True
                               )
                ],
                random_order=True
            )

            self.seq2 = iaa.Sequential(
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
                        mode=["reflect", "symmetric"]  # use any of scikit-images's warping modes (see 2nd images from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per images
                    # don't execute all of them, as that would often be way too strong
                    iaa.WithChannels([0, 1, 2],
                                     iaa.SomeOf((0, 7),
                                                [
                                                    # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                                                    # convert images into their superpixel representation
                                                    iaa.OneOf([
                                                        iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                                        iaa.AverageBlur(k=(2, 7)),
                                                        # blur images using local means with kernel sizes between 2 and 7
                                                        iaa.MedianBlur(k=(3, 11)),
                                                        # blur images using local medians with kernel sizes between 2 and 7
                                                    ]),
                                                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images  # { REMOVED AS NOT WORKING ON MULTIPROCESSING https://github.com/aleju/imgaug/issues/147
                                                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                                                    #search either for all edges or for directed edges,
                                                    #blend the result with the original images using a blobby mask
                                                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                                                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                                                    ])),                                                                  # }
                                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                                    # add gaussian noise to images
                                                    iaa.OneOf([
                                                        iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                                                    ]),
                                                    iaa.Invert(0.05, per_channel=True),  # invert color channels
                                                    iaa.Add((-10, 10), per_channel=0.5),
                                                    # change brightness of images (by -10 to 10 of original value)
                                                    iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                                                    # either change the brightness of the whole images (sometimes
                                                    # per channel) or change the brightness of subareas
                                                    iaa.OneOf([
                                                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                                        iaa.FrequencyNoiseAlpha(
                                                            exponent=(-4, 0),
                                                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                                            second=iaa.ContrastNormalization((0.5, 2.0))
                                                        )
                                                    ]),
                                                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                                                    #iaa.Grayscale(alpha=(0.0, 1.0)),
                                                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                                                    # move pixels locally around (with random strengths)
                                                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                                                    # sometimes move parts of the images around
                                                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                                                ],
                                                random_order=True
                                                ))
                ],
                random_order=True
            )


        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        assert(self.opt.loadSize >= self.opt.fineSize)
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        if self.opt.isTrain:
            A = np.array(A).astype(np.uint8)
            B = np.array(B).astype(np.uint8)
            A, B = self.augment_input(A, B, level=self.opt.augment_level)
            if self.opt.Atype == "seg_map":
                A = self.split_gt(A, merge_cls={160: 200})
        A = transforms.ToTensor()(A.copy()).float()
        B = transforms.ToTensor()(B.copy()).float()
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        if not self.opt.Atype == "seg_map":
            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

    def augment_input(self, gt, img, level=0):
        """
        my func to augment data further
        """
        if gt.ndim < img.ndim:
            gt = gt[:, :, np.newaxis]
        cat = np.concatenate((img, gt), axis=2)
        cat = eval("self.seq{}".format(level)).augment_image(cat)
        img = cat[:, :, 0:self.opt.input_nc]
        gt = cat[:, :, self.opt.input_nc:self.opt.input_nc + self.opt.output_nc]
        self.cnt += 1
        return gt, img


def split_gt(gt, cls_values=(0, 160, 200, 250), merge_cls=None, threshold=0.08):
    cls_gts = []
    if gt.ndim == 3:
        gt = gt[..., 0]

    # Build one gt images per class
    for c in cls_values:
        classmap = np.array(np.isclose(gt, c, atol=25), dtype=np.uint8)  # simple 0 or 1 for different classes
        cls_gts.append(classmap)

    # Zero out map if there are too few values compared to total object area (get rid of border artifacts)
    total_segment_area = float(cls_gts[0].shape[0] * cls_gts[0].shape[1] - np.sum(cls_gts[0]))
    for c in range(1, len(cls_gts) - 1):
        class_segment_area = np.sum(cls_gts[c])
        if total_segment_area < 10:
            cls_gts[c][...] = 0
        elif class_segment_area / total_segment_area < threshold:
            cls_gts[-1][cls_gts[c]] = 1  # add to lumen mask
            cls_gts[c] = np.zeros(cls_gts[c].shape)  # zero out artifacts

    # Handle overlapping classes (fill)
    if merge_cls:
        for cs, ct in merge_cls.items():
            mask = cls_gts[cls_values.index(cs)] > 0
            cls_gts[cls_values.index(ct)][mask] = 1  # fill with value
            del cls_gts[cls_values.index(cs)]

    gt2 = np.stack(cls_gts, axis=2).astype(np.float16)
    gt2[gt2 == 0] = -1  # output of network is tanh (-1, 1)
    return gt2