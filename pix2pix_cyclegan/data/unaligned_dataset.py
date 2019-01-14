import os.path
from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset
from PIL import Image
import random
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
import cv2

class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

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
                    mode="reflect"  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
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
                               iaa.WithChannels([0, 1, 2], iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),  # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.WithChannels([0, 1, 2], iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ]))),
                               iaa.WithChannels([0, 1, 2],
                                                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
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

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.opt.isTrain:
            A_img = np.array(A_img).astype(np.uint8)
            B_img = np.array(B_img).astype(np.uint8)
            if self.opt.direction == 'AtoB':
                A_img, B_img = self.augment_input(A_img, B_img)
            else:
                B_img, A_img = self.augment_input(B_img, A_img)
            A_img = Image.fromarray(A_img)
            B_img = Image.fromarray(B_img)

        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

    def augment_input(self, gt, img):
        if gt.ndim < img.ndim:
            gt = gt[:,:,np.newaxis]
        #img = cv2.bilateralFilter(img, 3, 75, 75)  # fails with seg error ?
        cat = np.concatenate((img, gt), axis=2)
        cat = self.seq.augment_image(cat)
        gt = cat[:,:,3:6]
        img = cat[:,:,0:3]
        return gt, img
