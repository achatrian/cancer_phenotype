from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into an images array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, segmap=False, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()  # taking the first images only
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    if segmap:
        image_numpy = segmap2img(image_numpy)
    else:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        # for segmentation maps with four classes
    return image_numpy.astype(imtype)


def segmap2img(segmap):
    """
    for 4 classes only
    output of network is tanh (-1, 1), so probability = 0.5 is taken as output = 0.0
    """
    if segmap.shape[0] == 4:
        segmap = segmap.transpose(1, 2, 0)
        image = np.argmax(segmap, axis=2)
        image[image == 1] = 160
        image[image == 2] = 200
        image[image == 3] = 250
    elif segmap.shape[0] == 3:
        segmap = segmap.transpose(1, 2, 0)
        image = np.argmax(segmap, axis=2)
        image[image == 1] = 200
        image[image == 2] = 250
    image = image[:, :, np.newaxis].repeat(3, axis=2)
    return image


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
