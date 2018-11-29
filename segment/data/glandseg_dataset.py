import os
import glob
import re
import cv2
from pathlib import Path
from itertools import product
import imageio
import torch
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import warnings
from base.data.base_dataset import BaseDataset, get_augment_seq, RandomCrop
from base.utils.utils import is_pathname_valid

ia.seed(1)


class GlandSegDataset(BaseDataset):

    def __init__(self, opt):
        super(GlandSegDataset, self).__init__()
        self.opt = opt

        self.file_list = []
        self.label = []

        phase_dir = os.path.join(self.opt.data_dir, self.opt.phase)
        folders = [name for name in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, name))]

        file_list = []
        for file in folders:
            file_list += glob.glob(os.path.join(phase_dir, file, 'tiles', '*_img_*.png'))

        self.file_list = file_list
        self.label = [x.replace('_img_', '_mask_') for x in file_list]

        self.randomcrop = RandomCrop(self.opt.crop_size)

        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)

    def __len__(self):
        return len(self.file_list)

    def name(self):
        return "GlandSegDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--segment_lumen', action='store_true')
        return parser

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        gt_name = self.label[idx]

        bgr_img = cv2.imread(img_name, -1)
        b, g, r = cv2.split(bgr_img)  # get b,g,r

        image = cv2.merge([r, g, b])  # switch it to rgb
        gt = cv2.imread(gt_name, -1)
        if not (isinstance(gt, np.ndarray) and gt.ndim > 0):
            raise ValueError("{} is not valid".format(gt_name))

        if gt.ndim == 3 and gt.shape[2] == 3:
            gt = gt[..., 0]
        if not self.opt.segment_lumen:
            gt[gt > 0] = 255

        if image.shape[0:2] != (self.opt.crop_size,)*2:
            too_narrow = image.shape[1] < self.opt.crop_size
            too_short = image.shape[0] < self.opt.crop_size
            if too_narrow or too_short:
                delta_w = self.opt.crop_size - image.shape[1] if too_narrow else 0
                delta_h = self.opt.crop_size - image.shape[0] if too_short else 0
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
                gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_REFLECT)

            if image.shape[0] > self.opt.crop_size or image.shape[1] > self.opt.crop_size:
                cat = np.concatenate([image, gt[:, :, np.newaxis]], axis=2)
                cat = self.randomcrop(cat)
                image = cat[:, :, 0:3]
                gt = cat[:, :, 3]

            # scale image
            sizes = (self.opt.fine_size, ) * 2
            image = cv2.resize(image, sizes, interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, sizes, interpolation=cv2.INTER_AREA)

        # im aug
        cat = np.concatenate([image, gt[:, :, np.newaxis]], axis=2)
        if self.opt.augment_level:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # but future extensions could be causing problems
                cat = self.aug_seq.augment_image(cat)
        image = cat[:, :, 0:3]
        gt = cat[:, :, 3]
        if not self.opt.segment_lumen:
            gt[gt < 255] = 0
            gt[gt != 0] = 1
        else:
            gt[np.logical_and(gt < 180, gt > 30)] = 1
            gt[gt >= 180] = 2
            gt[np.logical_and(gt != 1, gt != 2)] = 0

        # scale between 0 and 1
        image = image/255.0
        # normalised image between -1 and 1
        image = (image - 0.5)/0.5

        # convert to torch tensor
        assert(image.shape[-1] == 3)
        assert(len(gt.shape) == 2)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        gt = torch.from_numpy(gt.copy()).long()  # change to FloatTensor for BCE
        return {'input': image, 'target': gt}


class AugDataset(GlandSegDataset):

    def __init__(self, dir_, aug_dir, mode, tile_size, augment=0, generated_only=False):
        super(AugDataset, self).__init__()
        if generated_only:
            self.file_list = []
            self.label = []
        n = "[0-9]"
        names = ["_rec1_", "_rec2_"] #, "_gen1_", "_gen2_"]
        no_aug_len = len(self.file_list)
        file_list = []
        for gl_idx, x, y, name in product(range(1, 7), range(1, 6), range(1, 6), names):
            to_glob = os.path.join(aug_dir, 'gland_img_' + n * gl_idx + '_(' + n * x + ',' + n * y + ')' + \
                                   name + 'fake_B.png')
            file_list += glob.glob(to_glob)
        self.file_list += file_list
        self.label += [x.replace('fake_B', 'real_A') for x in file_list]
        assert (len(self.file_list) > no_aug_len)


class TestDataset(GlandSegDataset):
    def __init__(self, dir_, tile_size=256, bad_folds=[]):
        super(TestDataset, self).__init__()
        if bad_folds:
            for image_file, label_file in zip(self.file_list, self.label):
                image_name = os.path.basename(image_file)
                label_name = os.path.basename(label_file)
                assert(image_name[0:10] == label_name[0:10])
                isbad = any([bad_fold in image_name for bad_fold in bad_folds])
                if isbad:
                    self.file_list.remove(image_file)
                    self.label.remove(label_file)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(phase="test")
        return parser


class ProstateDataset(BaseDataset):

    def __init__(self, dir, mode, out_size=1024, down=2.0, num_class=1, grayscale=False, augment=None, load_wm=False):
        r"""
        Dataset to return random (pre-made) tiles of WSI image at original resolution.
        Can downscale
        Make grayscale
        Augment
        Use pre-made weight map for each example
        """
        self.mode = mode
        self.dir = dir
        self.out_size = out_size
        self.down = down
        self.num_class = num_class
        self.grayscale = grayscale
        self.augment = augment #set augment, or augment when training
        self.load_wm = load_wm
        assert(mode in ['train', 'validate', 'test'])

        #Read data paths
        self.gt_files = glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9],[0-9].png'), recursive=True) #for 1,1 patch
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9][0-9],[0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9][0-9][0-9],[0-9][0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9][0-9][0-9][0-9],[0-9][0-9][0-9][0-9].png'), recursive=True))

        self.img_files = [re.sub('mask', 'img', gtfile) for gtfile in self.gt_files]
        if self.load_wm:
            self.wm_files = []
            for gtfile in self.gt_files:
                srch = re.search('_mask_([0-9\(\)]+),([0-9\(\)]+).png', gtfile)
                self.wm_files.append(str(Path(gtfile).parents[1]/"weightmaps"/"weightmap_{},{}.png".format(srch.group(1), srch.group(2))))

        #Check paths
        path_check = zip(self.gt_files, self.img_files, self.wm_files) if self.load_wm else \
                        zip(self.gt_files, self.img_files)
        for idx, paths in enumerate(path_check):
            for path in paths:
                if not is_pathname_valid(path):
                    warnings.warn("Invalid path {} was removed".format(self.gt_files[idx]))
                    del self.gt_files[idx]


        assert(self.gt_files); r"Cannot be empty"
        assert(self.img_files);

        #Augmentation sequence
        self.downsample = Downsample(down, min_size=out_size)
        if self.mode == 'train': self.crop_aug = RandomCrop(int(out_size*down))
        elif self.mode in ['validate', 'test']: self.center_crop = CenterCrop(int(out_size*down))

        if self.augment:
            geom_augs = [iaa.Affine(rotate=(-45, 45)),
                        iaa.Affine(shear=(-10, 10)),
                        iaa.Fliplr(0.9),
                        iaa.Flipud(0.9),
                        iaa.PiecewiseAffine(scale=(0.01, 0.04)),
                        ]
            img_augs = [iaa.AdditiveGaussianNoise(scale=0.05*255),
                            iaa.Add(20, per_channel=True),
                            iaa.Dropout(p=(0, 0.2)),
                            iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.25),
                            iaa.AverageBlur(k=(2, 5)),
                            iaa.MedianBlur(k=(3, 5)),
                            iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.9, 1.1)),
                            iaa.Emboss(alpha=(0.0, 0.3), strength=(0.1, 0.3)),
                            iaa.EdgeDetect(alpha=(0.0, 0.3))]
            if not self.grayscale:
                #Add color modifications
                img_augs.append(iaa.WithChannels(0, iaa.Add((5, 20))))
                img_augs.append(iaa.WithChannels(1, iaa.Add((5, 10))))
                img_augs.append(iaa.WithChannels(2, iaa.Add((5, 20))))
                #img_augs.append(iaa.ContrastNormalization((0.8, 1.0)))
                img_augs.append(iaa.Multiply((0.9, 1.1)))
                #img_augs.append(iaa.Invert(0.5))
                img_augs.append(iaa.Grayscale(alpha=(0.0, 0.2)))
            self.geom_aug_seq = iaa.SomeOf((0, None), geom_augs) #apply 0-all augmenters; both img and gt
            self.img_seq = iaa.SomeOf((0, None), img_augs) #only img

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            img = imageio.imread(self.img_files[idx])
        except ValueError as err:
            print("#--------> Invalid img data for path: {}".format(self.img_files[idx]))
            raise err
        try:
            gt = imageio.imread(self.gt_files[idx])
        except ValueError as err:
            print("#--------> Invalid gt data for path: {}".format(self.img_files[idx]))
            raise err
        if self.load_wm:
            try:
                wm = imageio.imread(self.wm_files[idx]) / 255 #binarize
                wm = np.expand_dims(wm[:,:,0], 2)
            except ValueError as err:
                print("#--------> Invalid wm data for path: {}".format(self.img_files[idx]))
                raise err
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
            #if self.load_wm:
                #wm = cv2.copyMakeBorder(wm, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            assert(img.shape[0] >= size_b4_down and img.shape[1] >= size_b4_down)  #Ensure padded
            assert(gt.shape[0] >= size_b4_down and gt.shape[1] >= size_b4_down)  #Ensure padded

        #Transform gt to desired number of classes
        if self.num_class > 1: gt = self.split_gt(gt)
        else: gt=(gt>0)[:,:,np.newaxis].astype(np.uint8)

        #Grascale
        if self.grayscale: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.mode == 'train':
            img = self.crop_aug([img], random_state=1)[0] #crop to out_size * downsample
            img = self.downsample(img) #downsample to size out_sizee
            gt = self.crop_aug([gt], random_state=1)[0]
            gt = self.downsample(gt)
            #if self.load_wm:
                #wm = self.crop_aug([wm], random_state=1)[0]
                #wm = self.downsample(wm)
        else:
            img = self.center_crop(img) #center crop tiles when testing
            img = self.downsample(img) #downsample like in training
            gt = self.center_crop(gt)
            gt = self.downsample(gt)
            #if self.load_wm:
                #wm = self.center_crop([wm])[0]
                #wm = self.downsample(wm)

        if self.augment:
            geom_seq_det = self.geom_aug_seq.to_deterministic() #ensure ground truth and image are transformed identically
            img = geom_seq_det.augment_image(img)
            img = self.img_seq.augment_image(img)
            img.clip(0, 255) #ensure added values don't stray from normal boundaries
            gt = geom_seq_det.augment_image(gt)
            if self.load_wm:
                wm = geom_seq_det.augment_image(wm)
            #print("THE SHAPE OF WM IS {}".format(wm.shape))

        example = self.to_tensor(img, isimage=True),  self.to_tensor(gt, isimage=False)
        if self.load_wm:
            example += (self.to_tensor(wm, isimage=False),)
        return example

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

    def to_tensor(self, na, isimage):
        r"""Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        na = na[np.newaxis,:,:] if self.grayscale and isimage else na.transpose((2, 0, 1)) #grayscale or RGB
        na = torch.from_numpy(na.copy()).type(torch.FloatTensor)
        return na

