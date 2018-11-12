import os
import argparse
from pathlib import Path
from numbers import Integral
from torchvision import transforms
import numpy as np
import scipy.signal
import gc
import torch
from torch import load
import imageio
from models import UNet1, UNet2, UNet3, UNet4
import time
from PIL import Image
from torch import cuda
from torch.nn import DataParallel
import cv2
import random
from PIL import ImageEnhance
from utils import get_flags

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()
parser.add_argument('--save_folder', default='/well/win/users/achatrian/cancer_phenotype/Results', type=str,
                    help='Dir to save results')
parser.add_argument('--image', default='/well/win/users/achatrian/cancer_phenotype/Dataset/train/17_A047-4463_153D+-+2017-05-11+09.40.22_TissueTrain_(1.00,35171,30052,4796,3922)/17_A047-4463_153D+-+2017-05-11+09.40.22_TissueTrain_(1.00,35171,30052,4796,3922)_img.png', type=str)
parser.add_argument('--checkpoint_folder', default='/well/win/users/achatrian/cancer_phenotype/logs/2018_09_17_20_56_36/ckpt', type=str, help='checkpoint folder')
parser.add_argument('--snapshot', default='epoch_.171_loss_0.21296_acc_0.93000_dice_0.93000_lr_0.0000686500.pth', type=str)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')
parser.add_argument('--batch_size', default=8, type=int)

parser.add_argument('--num_class', type=int, default=1)
parser.add_argument('-nf', '--num_filters', type=int, default=37, help='mcd number of filters for unet conv layers')
parser.add_argument('--network_id', type=str, default="UNet3")

FLAGS = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

if not os.path.exists(FLAGS.save_folder):
    os.mkdir(FLAGS.save_folder)

ckpt_path = Path(FLAGS.checkpoint_folder)  # scope? does this change inside train and validate functs as well?
#LOADEDFLAGS = get_flags(str(ckpt_path / ckpt_path.parent.name) + ".txt")
#FLAGS.num_filters = LOADEDFLAGS.num_filters
#FLAGS.num_class = LOADEDFLAGS.num_class

exp_name=''

########################################################################################

class CustomisedTransform(object):
    # hue
    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        # invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def __call__(self, img):
        r, g, b = cv2.split(img)

        rr = np.random.uniform(low=np.log(0.25), high=np.log(4))
        rb = np.random.uniform(low=np.log(0.25), high=np.log(4))
        r = self.adjust_gamma(r, gamma=np.exp(rr))
        b = self.adjust_gamma(b, gamma=np.exp(rb))
        image = cv2.merge([r, g, b])  # switch it to rgb

        image = Image.fromarray(image)

        # brightness
        brightness = 0.4
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        image = enhancer.enhance(brightness_factor)

        # contrast
        contrast = 0.4
        contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        # saturation
        saturation = 0.4
        saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)

        # convert image back to numpy array
        image = np.array(image)

        # flip horizontal
        if random.random() < 0.5:
            image = cv2.flip(image, 1).reshape(image.shape)

        # flip vertical
        if random.random() < 0.5:
            image = cv2.flip(image, 0).reshape(image.shape)

        # flip transpose
        if random.random() < 0.5:
            image = cv2.flip(image, -1).reshape(image.shape)

        return image


########################################################################################
class TilePrediction(object):
    def __init__(self, patch_size, subdivisions, scaling_factor, pred_model, batch_size):
        """
        :param patch_size:
        :param subdivisions: the size of stride is define by this
        :param scaling_factor: what factor should prediction model operate on
        :param pred_model: the prediction function
        """
        self.patch_size = patch_size
        self.scaling_factor = scaling_factor
        self.subdivisions = subdivisions
        self.pred_model = pred_model
        self.batch_size = batch_size

        self.stride = int(self.patch_size / self.subdivisions)

        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

        self.WINDOW_SPLINE_2D = self._window_2D(window_size=self.patch_size, effective_window_size=patch_size, power=2)

    def _read_data(self, filename):
        """
        :param filename:
        :return:
        """
        img = cv2.imread(filename, -1)
        img = img[..., ::-1]

        img = cv2.resize(img, (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)
        img = self._pad_img(img)

        return img

    def _pad_img(self, img):
        """
        Add borders to img for a "valid" border pattern according to "window_size" and
        "subdivisions".
        Image is an np array of shape (x, y, nb_channels).
        """
        aug = int(round(self.patch_size * (1 - 1.0 / self.subdivisions)))
        more_borders = ((aug, aug), (aug, aug), (0, 0))
        ret = np.pad(img, pad_width=more_borders, mode='reflect')

        return ret

    def _unpad_img(self, padded_img):
        """
        Undo what's done in the `_pad_img` function.
        Image is an np array of shape (x, y, nb_channels).
        """
        aug = int(round(self.patch_size * (1 - 1.0 / self.subdivisions)))
        ret = padded_img[aug:-aug, aug:-aug, :]
        return ret

    def _spline_window(self, patch_size, effective_window_size, power=2):
        """
        Squared spline (power=2) window function:
        https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
        """
        window_size = effective_window_size
        intersection = int(window_size / 4)
        wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)

        aug = int(round((patch_size - window_size) / 2.0))
        wind = np.pad(wind, (aug, aug), mode='constant')
        wind = wind[:patch_size]

        return wind

    def _window_2D(self, window_size, effective_window_size, power=2):
        """
        Make a 1D window function, then infer and return a 2D window function.
        Done with an augmentation, and self multiplication with its transpose.
        Could be generalized to more dimensions.
        """
        # Memoization
        wind = self._spline_window(window_size, effective_window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 3), 3)
        wind = wind * wind.transpose(1, 0, 2)
        return wind

    def _extract_patches(self, img):
        """
        :param img:
        :return: a generator
        """
        step = int(self.patch_size / self.subdivisions)

        row_range = range(0, img.shape[0] - self.patch_size + 1, step)
        col_range = range(0, img.shape[1] - self.patch_size + 1, step)

        for row in row_range:
            for col in col_range:
                x = img[row:row + self.patch_size, col:col + self.patch_size, :]
                x = self.transform(x)

                yield x

    def _merge_patches(self, patches, padded_img_size):
        """
        :param patches:
        :param padded_img_size:
        :return:
        """
        n_dims = patches[0].shape[-1]
        img = np.zeros([padded_img_size[0], padded_img_size[1], n_dims], dtype=np.float32)

        window_size = self.patch_size
        step = int(window_size / self.subdivisions)

        row_range = range(0, img.shape[0] - self.patch_size + 1, step)
        col_range = range(0, img.shape[1] - self.patch_size + 1, step)

        for index1, row in enumerate(row_range):
            for index2, col in enumerate(col_range):
                tmp = patches[(index1 * len(col_range)) + index2]
                # tmp = tmp[np.newaxis, np.newaxis, :]
                # tmp = np.repeat(tmp, self.patch_size, axis=0)
                # tmp = np.repeat(tmp, self.patch_size, axis=1)
                tmp *= self.WINDOW_SPLINE_2D
                # tmp = (np.ones((self.patch_size, self.patch_size, n_dims), dtype=np.float32) * tmp) * self.WINDOW_SPLINE_2D

                img[row:row + self.patch_size, col:col + self.patch_size, :] = \
                    img[row:row + self.patch_size, col:col + self.patch_size, :] + tmp

        img = img / (self.subdivisions ** 2)
        return self._unpad_img(img)

    def batches(self, generator, size):
        """
        :param generator: a generator
        :param size: size of a chunk
        :return:
        """
        source = generator
        while True:
            chunk = [val for _, val in zip(range(size), source)]
            if not chunk:
                raise StopIteration
            yield chunk

    def run(self, filename):
        """
        :param filename:
        :return:
        """

        # read image, scaling, and padding
        padded_img = self._read_data(filename)

        # extract patches
        patches = self._extract_patches(padded_img)
        gc.collect()

        # run the model in batches
        all_prediction = []

        for ipatch, chunk in enumerate(self.batches(patches, self.batch_size)):
            x = []

            for ch in chunk:
                x.append(ch.unsqueeze(0))

            img = torch.cat(x, dim=0)

            all_prediction += [self.pred_model(img).cpu().data.numpy()]

        all_prediction = np.concatenate(all_prediction, axis=0)
        all_prediction = all_prediction.transpose(0, 2, 3, 1)

        result = self._merge_patches(all_prediction, padded_img.shape)

        # confidence
        uncertain = ((np.logical_and(result < 0.8, result > 0.3)) * 255.0).astype(np.uint8)

        result = (result > 0.55) * 255.0
        result = result.astype(np.uint8)
        result = cv2.resize(result, (0, 0), fx=1.0 / self.scaling_factor, fy=1.0 / self.scaling_factor)

        uncertain = cv2.resize(uncertain, (0, 0), fx=1.0 / self.scaling_factor, fy=1.0 / self.scaling_factor)

        result = np.clip(result, 0, 255).astype(np.uint8)
        uncertain = np.clip(uncertain, 0, 255).astype(np.uint8)

        return result, uncertain


def main():
    if cuda.is_available():
        cuda.set_device(0)

    # check if the input image exists
    if os.path.exists(FLAGS.image):

        # basename
        basename = os.path.basename(FLAGS.image)
        basename = os.path.splitext(basename)[0]
        savename = os.path.join(FLAGS.save_folder, basename + '.png')
        savename_conf = os.path.join(FLAGS.save_folder, basename + '_confidence.png')

        if not os.path.exists(savename):

            parallel = True
            inputs = {'num_classes': FLAGS.num_class, 'num_channels': FLAGS.num_filters}
            if FLAGS.network_id == "UNet1":
                net = UNet1(**inputs).cuda() if cuda.is_available() else UNet1(**inputs)
            elif FLAGS.network_id == "UNet2":
                net = UNet2(**inputs).cuda() if cuda.is_available() else UNet2(**inputs)
            elif FLAGS.network_id == "UNet3":
                net = UNet3(**inputs).cuda() if cuda.is_available() else UNet3(**inputs)
            elif FLAGS.network_id == "UNet4":
                net = UNet4(**inputs).cuda() if cuda.is_available() else UNet4(**inputs)
            if parallel:
                net = DataParallel(net).cuda()

            print('Load model: ' + FLAGS.snapshot)

            dev = None if cuda.is_available() else 'cpu'
            state_dict = load(str(ckpt_path / exp_name / FLAGS.snapshot), map_location=dev)
            if not parallel:
                state_dict = {key[7:]: value for key, value in state_dict.items()}
            net.load_state_dict(state_dict)

            net.eval()
            predictor = TilePrediction(patch_size=512,
                                       subdivisions=2.0,
                                       scaling_factor=0.5,
                                       pred_model=net,
                                       batch_size=6)

            segmentation, uncertain = predictor.run(FLAGS.image)

            imageio.imwrite(savename, segmentation)
            imageio.imwrite(savename_conf, uncertain)


#############################################################################################

if __name__ == '__main__':
    t = time.time()
    main()
    print('finish image (%.2f)' % (time.time() - t))
