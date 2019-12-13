from pathlib import Path
import argparse
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd
import numpy as np
from skimage import color
from quant.experiment import BaseExperiment
from data.images.wsi_reader import WSIReader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=Path, default=Path('~/Documents/Temp/projector_glands_logs').expanduser())
    parser.add_argument('--data_dir', type=Path, default='/mnt/rescomp/projects/prostate-gland-phenotyping/WSI')
    parser.add_argument('--memory_use', type=float, default=0.0)
    args = parser.parse_args()
    args.log_dir.makedirs(exist_ok=True)
    metadata_path = args.log_dir/'metadata.tsv'
    exp = BaseExperiment('projector', [], [])
    exp.read_data(args.data_dir/'data'/'features'/'all.h5')
    index = exp.x.index.levels[0] if isinstance(exp.x.index, pd.core.index.MultiIndex) else list(exp.x.index)
    bounding_boxes = list(tuple(int(d) for d in s.split('_')) for s in exp.x.index)
    images, slide_labels = [], []
    for slide_path in args.data_dir.iterdir():
        if slide_path.suffix not in ('.ndpi', '.svs'):
            continue
        slide = WSIReader(file_name=str(slide_path))
        slide_id = slide_path.with_suffix('').name
        x_slide = exp.x[slide_id]
        bounding_boxes = tuple(tuple(d for d in s.split('_')) for s in list(x_slide.index))
        slide_images = list(np.array(slide.read_region((x, y), 0, (w, h))) for x, y, w, h in bounding_boxes)
        for i, slide_image in slide_images:
            if slide_image.shape[2] == 4:
                slide_images[i] = color.rgba2rgb(slide_image)
        images.extend(slide_images)
        slide_labels.extend()
    images = tf.Variable(images, name='images')

    with tf.Session() as sess:
        saver = tf.train.Saver([images])
        sess.run(images.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))
        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = images.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = metadata
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)


def make_thumbnail(img, gt, th_size, label=None):
    """
    :param img:
    :param th_size:
    :param label:
    :return: thumbnail

    Generate thumbnail of required size given an images, taking into account different gland sizes
    If label is given, colour the background with the label of specific classes
    """

    img = (img + 1) / 2 * 255  # change back to RGB from [-1, 1] range
    assert img.min() > 0.0 - 0.1 and img.max() < 255.0 + 0.1
    img = img.astype(np.uint8)
    img_mode = [int(mode(img[np.logical_not(gt), 0], axis=None)[0]),
                int(mode(img[np.logical_not(gt), 1], axis=None)[0]),
                int(mode(img[np.logical_not(gt), 2], axis=None)[0])]
    background = int(label) or 0
    if label == 2:
        img[img[..., 0] == img_mode[0], 0] = 100
        img[img[..., 1] == img_mode[1], 1] = 170
        img[img[..., 2] == img_mode[2], 0] = 100
    elif label == 3:
        img[img[..., 0] == img_mode[0], 0] = 100
        img[img[..., 1] == img_mode[1], 1] = 100
        img[img[..., 2] == img_mode[2], 2] = 170
    else:
        pass
    seg = np.logical_not(np.isclose(background, img)).any(axis=2).astype(np.uint8)  # isclose to compare floats
    gt2, contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    gland_cnt = contours[areas.index(max(areas))]
    x, y, w, h = cv2.boundingRect(gland_cnt)
    s = max(w, h)
    s = min(int(round(s * 1.5)), img.shape[0])  # increase bounding box for better visualization
    x = min(x, img.shape[0] - s)  # img is a square
    y = min(y, img.shape[0] - s)
    tn = img[y: y + s, x: x + s]
    # tn = np.array(Image.fromarray(tn).thumbnail(th_size, Image.ANTIALIAS)) # could upsample or downsample depending on original gland size
    tn = cv2.resize(tn, th_size, cv2.INTER_AREA)
    return tn.astype(np.uint8)


def make_thumbnail_fullcut(img, gt, th_size, label=None):
    """
    :param img:
    :param th_size:
    :param label:
    :return: thumbnail

    Generate thumbnail of required size given an images, taking into account different gland sizes
    If label is given, colour the background with the label of specific classes
    """

    img = (img + 1) / 2 * 255  # change back to RGB from [-1, 1] range
    assert img.min() > 0.0 - 0.1 and img.max() < 255.0 + 0.1
    img = img.astype(np.uint8)
    background = int(label) or 0
    kernel = np.ones((8, 8), np.uint8)
    gt2 = cv2.erode(np.uint8(gt > 0), kernel, iterations=1)
    gt2, contours, hierarchy = cv2.findContours(gt2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cX, cY = 0, 0
    center = (img.shape[0] / 2, img.shape[1] / 2)
    best_dist = abs(cX - center[1]) + abs(cY - center[0])

    # Find central gland
    for contour in contours:
        M = cv2.moments(contour)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            pass

        dist_new = abs(cX - center[1]) + abs(cY - center[0])
        if dist_new < best_dist:
            best_dist = dist_new
            gland_contour = contour

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [gland_contour], -1, 1, -1)
    mask = cv2.dilate(mask, kernel, iterations=2)  # dilate to increase again (2 iters)
    img = cv2.bitwise_and(img, img, mask=mask)

    # Colour background with label
    mask = np.logical_not(mask)
    if background == 2:
        img[mask, 0] = 100
        img[mask, 1] = 170
        img[mask, 0] = 100
    elif background == 1:
        img[mask, 0] = 100
        img[mask, 1] = 100
        img[mask, 2] = 170
    else:
        pass

    # Make thumbnail
    x, y, w, h = cv2.boundingRect(gland_contour)
    s = max(w, h)
    s = min(int(round(s * 1.3)), img.shape[0])  # increase bounding box for better visualization
    x = min(x, img.shape[0] - s)  # img is a square
    y = min(y, img.shape[0] - s)
    tn = img[y: y + s, x: x + s]
    tn = cv2.resize(tn, th_size, cv2.INTER_AREA)
    return tn.astype(np.uint8)


def create_sprite_image(images):
    """Returns a sprite images consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots, 3))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img
    return spriteimage


def write_metadata(filename, labels):
    """
            Create a metadata file images consisting of sample indices and labels
            :param filename: name of the file to save on disk
            :param shape: tensor of labels
    """
    with open(filename, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("{}\t{}\n".format(index, label))

    print('Metadata file saved in {}'.format(filename))
