import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


if __name__ == '__main__':
    LOG_DIR = '~/Documents/Temp/projector_logs'
    os.makedirs(LOG_DIR, exist_ok=True)
    metadata = os.path.join(LOG_DIR, 'metadata.tsv')

    mnist = input_data.read_data_sets('MNIST_data')
    images = tf.Variable(mnist.test.images, name='images')

    with open(metadata, 'w') as metadata_file:
        for row in mnist.test.labels:
            metadata_file.write('%d\n' % row)

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


# Make objects for feature extraction
# orb = cv2.ORB_create()  # ORB features

kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 5):
        for frequency in (0.05, 0.25):
            # Gabor patches
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernel = np.tile(kernel[...,np.newaxis], (1,1,3))
            kernels.append(kernel)


def feature_extraction(imgs, gts):
    """
    :param imgs:
    :param gts:
    :return:

    Extract features:
    SIFT
    ORB > SIFT
    SHAPE
    FILTER BANK RESPONSE
    """

    features = []
    for n, (img, gt) in enumerate(zip(imgs, gts)):
        gt = gt.squeeze()
        if not gt.any():
            gt = np.ones(gt.shape, dtype=np.bool)

        # kp = orb.detect(img, None)
        # kp, orb_descriptors = orb.compute(img, kp)

        kernel_descriptors = []
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(img, kernel, mode='wrap')
            kernel_descriptors.extend((filtered[gt > 0].mean(), filtered[gt > 0].var()))

        haralick_feats = mahotas.features.haralick(img.astype(np.uint8), distance=2)
        haralick_feats = list(haralick_feats.flatten())

        props = regionprops(gt.astype(np.uint8), cache=True)[0]
        props_list = [props.equivalent_diameter, props.eccentricity, props.extent]
        props_list += list(props.moments_central.flatten())

        gland_feats = kernel_descriptors + haralick_feats + props_list
        features.append(gland_feats)
    assert features
    return np.array(features)


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


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

ia.seed(7)
alpha = 0.3
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.8),  # horizontally flip
        iaa.Flipud(0.8),  # vertically flip
        # crop images by -5% to 10% of their height/width
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.05, 0.1),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-2, 2),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-images's warping modes (see 2nd images from the top for examples)
        ),
    ],
    random_order=True
)
seq = seq.to_deterministic()  # to ensure that every images is augmented in the same way (so that features are more comparable)
def augment_glands(imgs, gts, N=1):
    """
    :param imgs:
    :param gts:
    :param N:
    :return: Original images + N augmented copies of images and ground truths
    """
    if N:
        M = imgs.shape[0]
        # Does nothing for N = 0
        imgs = (imgs + 1) / 2 * 255
        order = [m + M*n for m in range(M) for n in range(N+1)] # reorder images so that images for same gland are consecutive
        if gts.ndim < imgs.ndim:
            gts = gts[..., np.newaxis]
        cat = np.concatenate([imgs, gts], axis=3)
        cat = np.tile(cat, (N,1,1,1))
        out = seq.augment_images(cat)  # works on batch of images
        gts = np.concatenate((gts, out[..., 3:4]), axis=0)  # need to keep channel dim
        imgs = np.concatenate((imgs, out[..., 0:3]), axis=0)
        imgs, gts = imgs[order], gts[order]
        imgs = (imgs / 255 - 0.5)/0.5  # from -1 to 1
    return imgs, gts