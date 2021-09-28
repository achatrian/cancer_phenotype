from pathlib import Path
import datetime
import numpy as np
from matplotlib import pyplot as plt
import imageio


def show_image(image, title=None):
    # plot images quickly
    plt.imshow(image)
    if title is not None:
        ax = plt.gca()
        ax.set_title(title)
    plt.show()


def save_image(image, path=Path('/well/rittscher/users/achatrian/debug'), name=None):
    # save images when plotting in pycharm remotely doesn't work
    try:
        path.mkdir(exist_ok=True, parents=True)
        if not name:
            path = path / (str(datetime.datetime.now()) + '.png')  # add date to make unique
        else:
            path = path / (name + '.png')
        imageio.imsave(path, image)
    except (NotADirectoryError, IOError):
        path = Path('/mnt/rittscher/users/achatrian/debug')
        if not name:
            path = path / (str(datetime.datetime.now()) + '.png')  # add date to make unique
        else:
            path = path / (name + '.png')
        imageio.imsave(path, image)


def image_stats(image):
    image = np.array(image)
    print(f"Image shape: {image.shape}, dtype = {image.dtype} max = {image.max()}, min = {image.min()}, mean = {image.mean()}, std = {image.std()}")
    print(f"Unique values: {list(np.unique(image))}")


