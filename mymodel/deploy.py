import gc
import os, glob
import time
import torch
import argparse
import torch.utils.data
import scipy.signal
import pickle

import multiprocessing as mp
import torch.backends.cudnn as cudnn
import numpy as np
import lstm_cnn_data as lstm_cnn_dataset
import imagio

from torch.autograd import Variable
from lstm_cnn import LSTMCNN
from functools import partial

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

parser = argparse.ArgumentParser()

parser.add_argument('--result_dir', default='../temp/step3_tissue_segmentation', type=str)
parser.add_argument('--test_folder', default='../temp/step1_colour_norm', type=str)
parser.add_argument('--image_extension', default='.png', type=str)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--patch_size', default=512, type=int)
parser.add_argument('--test_stride', default=42, type=int)
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')

parser.add_argument('--learning_rate', default=0.0002, type=float)

parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--workers', default=1, type=int)
parser.add_argument('--checkpoint_folder', default='checkpoints', type=str)
parser.add_argument('--resume', default='checkpoints/checkpoint_46.pth', type=str)

parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
parser.add_argument('--nf', type=int, default=64)


class TileWorkersTissueSegmentation(mp.Process):
    def __init__(self, queue, predictor):
        mp.Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._predictor = predictor

    def run(self):
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break

            file, result_dir, wsi_name, i_tile, total_file = data[0], data[1], data[2], data[3], data[4]

            start_time = time.time()

            basename = os.path.basename(file)
            basename = os.path.splitext(basename)[0]
            savename = os.path.join(result_dir, wsi_name, basename + '.png')
            savename_confidence = os.path.join(result_dir, wsi_name, basename + '_confidence.png')

            if not os.path.exists(savename):
                # evaluation mode
                result, confidence = self._predictor.run(file)
                KSimage.imwrite(result, savename)
                KSimage.imwrite(confidence, savename_confidence)

            duration = time.time() - start_time

            print('Finish segmenting tissue regions on H&E tile %d / %d (%.2f sec)' % (i_tile, total_file, duration))

            self._queue.task_done()

########################################################################################
class TilePrediction(object):
    def __init__(self, patch_size, effective_window_size, subdivisions, scaling_factor, pred_model, batch_size):
        """
        :param patch_size:
        :param subdivisions: the size of stride is define by this
        :param scaling_factor: what factor should prediction model operate on
        :param pred_model: the prediction function
        """
        self.patch_size = patch_size
        self.effective_window_size = effective_window_size
        self.scaling_factor = scaling_factor
        self.subdivisions = subdivisions
        self.pred_model = pred_model
        self.batch_size = batch_size

        self.stride = int(self.patch_size/self.subdivisions)

        # scaling operation
        self.scale64 = lstm_cnn_dataset.Scale(64)
        self.scale128 = lstm_cnn_dataset.Scale(128)
        self.scale256 = lstm_cnn_dataset.Scale(256)
        self.crop = lstm_cnn_dataset.CenterCrop(64)

        self.WINDOW_SPLINE_2D = self._window_2D(window_size=self.patch_size, effective_window_size=self.effective_window_size, power=2)

    def _read_data(self, filename, scaling_factor):
        """
        :param filename:
        :return:
        """
        img = KSimage.imread(filename)
        img = KSimage.imresize(img, scaling_factor)

        if img.ndim == 2:
            img = np.expand_dims(img, axis=3)

        # padding
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
        window_size = 2*effective_window_size
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
                imgs = [self.scale64(x), self.scale128(x), self.scale256(x), x]
                imgs = [self.crop(x) for x in imgs]
                imgs = [x.transpose(2, 0, 1) / 255.0 for x in imgs]
                imgs = [torch.from_numpy(x).type(torch.FloatTensor) for x in imgs]

                yield (imgs)

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
                tmp = tmp[np.newaxis, np.newaxis, :]
                tmp = np.repeat(tmp, self.patch_size, axis=0)
                tmp = np.repeat(tmp, self.patch_size, axis=1)
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

    def _softmax(self, X, theta=1.0, axis=None):
        """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
            first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """

        # make X at least 2d
        y = np.atleast_2d(X)

        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

        # multiply y against the theta parameter,
        y = y * float(theta)

        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis=axis), axis)

        # exponentiate y
        y = np.exp(y)

        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

        # finally: divide elementwise
        p = y / ax_sum

        # flatten if X was 1D
        if len(X.shape) == 1:
            p = p.flatten()

        return p

    def run(self, filename):
        """
        :param filename:
        :return:
        """
        # read image, scaling, and padding
        padded_img = self._read_data(filename, self.scaling_factor)

        # extract patches
        patches = self._extract_patches(padded_img)
        gc.collect()

        # run the model in batches
        all_prediction = []
        for ipatch, chunk in enumerate(self.batches(patches, self.batch_size)):
            p0 = []
            p1 = []
            p2 = []
            p3 = []
            for ch in chunk:
                p0.append(ch[0].unsqueeze(0))
                p1.append(ch[1].unsqueeze(0))
                p2.append(ch[2].unsqueeze(0))
                p3.append(ch[3].unsqueeze(0))

            p0 = Variable(torch.cat(p0, dim=0).cuda(), requires_grad=False)
            p1 = Variable(torch.cat(p1, dim=0).cuda(), requires_grad=False)
            p2 = Variable(torch.cat(p2, dim=0).cuda(), requires_grad=False)
            p3 = Variable(torch.cat(p3, dim=0).cuda(), requires_grad=False)

            temp = [p0, p1, p2, p3]

            pred = self.pred_model(temp)
            all_prediction.append(pred.cpu().data.numpy())

            del temp
            del pred

        # merge patches and unpad
        gc.collect()
        all_prediction = np.concatenate(all_prediction, axis=0)
        tiled_prediction = self._merge_patches(all_prediction, padded_img.shape)

        # find prediction confidence
        prob = self._softmax(tiled_prediction, axis=2)
        prob = np.sort(prob, axis=2)
        confidence = prob[:, :, -1] - prob[:, :, -2]
        confidence = np.clip(confidence, 0, 1) * 255.0
        confidence = confidence.astype(np.uint8)

        # find maximum
        final_prediction = np.argmax(tiled_prediction, axis=2)
        final_prediction = final_prediction.astype(np.uint8)

        # rescale
        result = KSimage.imresize(final_prediction, 1/self.scaling_factor)
        confidence = KSimage.imresize(confidence, 1/self.scaling_factor)

        return result, confidence


########################################################################################
def load_model(args):
    """
    construct, load, and initialize the model
    :param args:
    :return:
    """
    # initialize CUDA
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if torch.cuda.is_available():
        torch.randn(8).cuda()

    # create a model
    net = LSTMCNN(args.image_size, 3, args.nf, args.nz, 4, args.learning_rate, args.batch_size)

    if torch.cuda.is_available():
        net.cuda()

    # load trained model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # set the model for evaluation
    net.eval()

    return net

####################################################################################
def main():
    global args
    # load input arguments
    args = parser.parse_args()

    # load the model
    net = load_model(args)
    net.share_memory()

    # create result folder
    routine.create_dir(args.result_dir)

    # list all folders
    wsi_list = [name for name in os.listdir(args.test_folder) if os.path.isdir(os.path.join(args.test_folder, name))]
    wsi_list.sort()

    # define tiled prediction operation
    predictor = TilePrediction(patch_size=args.patch_size,
                               effective_window_size=64,
                               subdivisions=12,
                               scaling_factor=0.5,
                               pred_model=net,
                               batch_size=128)

    for wsi_name in wsi_list:

        # list test image files
        filename_list = glob.glob(os.path.join(args.test_folder, wsi_name, '*' + args.image_extension))
        filename_list.sort()

        # create a directory
        routine.create_dir(os.path.join(args.result_dir, wsi_name))

        #########################################################
        num_workers = args.workers
        queue = mp.JoinableQueue(2 * num_workers)
        for i in range(num_workers):
            TileWorkersTissueSegmentation(queue, predictor).start()
        #########################################################

        # read list of data
        for iImage, file in enumerate(filename_list, 1):
            # start_time = time.time()
            #
            # basename = os.path.basename(file)
            # basename = os.path.splitext(basename)[0]
            # savename = os.path.join(args.result_dir, wsi_name, basename + '.png')
            #
            # if not os.path.exists(savename):
            #     # evaluation mode
            #     result = predictor.run(file)
            #
            #     KSimage.imwrite(result, savename)
            # duration = time.time() - start_time
            # print('Finish segmenting DCIS regions on the H&E image of sample %d out of %d samples (%.2f sec)' %
            #       (iImage + 1, len(filename_list), duration))

            queue.put((file, args.result_dir, wsi_name, iImage, len(filename_list)))

        ###############################################################
        for _i in range(num_workers):
            queue.put(None)
        queue.join()
        ###############################################################


if __name__ == '__main__':

    mp.set_start_method('spawn')
    from KS_lib.prepare_data import routine

    main()
