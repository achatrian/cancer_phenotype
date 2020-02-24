r"""Feature computations from contours, masks, and images"""
from inspect import getfullargspec
from pathlib import Path
import time
import warnings
from numbers import Real
import numpy as np
from skimage import measure
from skimage import color
from skimage import feature
import mahotas as mh
from mahotas.features import surf  # needs separate import or does not work
from base.utils import debug
from joblib import Memory
# caching_path = Path('~/python_caches').expanduser()
#memory = Memory(caching_path, bytes_limit=5e6, verbose=False)  # caching images so that checks are run faster
# caching_path.mkdir(exist_ok=True)


#@memory.cache
def is_contour(arg):  # specifies contour format
    return isinstance(arg, np.ndarray) and arg.ndim == 3 and arg.shape[2] == 2


#@memory.cache
def is_mask(arg, num_classes=0):  # specifies mask format
    return isinstance(arg, np.ndarray) and np.unique(arg).size <= (num_classes or 10) and arg.ndim == 2


#@memory.cache
def is_image(arg):  # specifies images format (RGB mapped to [-1, 1])
    return isinstance(arg, np.ndarray) and arg.max() <= 255 and arg.min() >= 0 and arg.ndim == 3 and arg.shape[2] == 3


#@memory.cache
def is_gray_image(arg):  # specifies gray images format (grayscale mapped to [-1, 1])
    return isinstance(arg, np.ndarray) and arg.max() <= 255 and arg.min() >= 0 and arg.ndim == 2


class Feature:
    r"""Feature callable"""
    __slots__ = ['function', 'type_', 'returns', 'name', 'call_time', 'n_calls', 'enable_checks',
                 'is_contour', 'is_mask', 'is_image', 'is_gray_image']
    # how-to-python: docstring and __slots__ defined this way are class attributes

    def __init__(self, function, returns, enable_checks=True):
        self.function = function
        type_ = set(getfullargspec(function).args)
        assert type_ >= {'contour'} or type_ >= {'mask'} or type_ >= {'image'} or type_ >= {'gray_image'}
        self.type_ = type_
        self.returns = returns
        self.name = function.__name__
        self.call_time = 0.0
        self.n_calls = 0
        self.enable_checks = enable_checks
        # TODO test memoized function attributes

    def __call__(self, **kwargs):
        # NB: only keyword arguments work with Features
        if set(kwargs) < self.type_:
            raise ValueError(f"Missing arguments {self.type_ - set(kwargs)} for feature '{self.name}'")
        if self.enable_checks:
            if 'contour' in self.type_ and not is_contour(kwargs['contour']):
                raise ValueError(f"Arguments do not contain contour-type input (f: {self.name})")
            if 'mask' in self.type_ and not is_mask(kwargs['mask']):
                raise ValueError(f"Arguments does not contain mask-type input (f: {self.name})")
            if 'image' in self.type_ and not is_image(kwargs['image']):
                raise ValueError(f"Arguments does not contain image-type input (f: {self.name})")
            if 'gray_image' in self.type_ and not is_gray_image(kwargs['gray_image']):
                raise ValueError(f"Arguments does not contain grayscale images-type input (f: {self.name})")
        start_time = time.time()
        output = self.function(**kwargs)
        # error if any output is nan or infinite
        if self.enable_checks and (np.isnan(output).any() or np.isinf(output).any()):
            invalid_features_names = tuple(self.returns[i] for i in np.where(output)[0])
            raise ValueError(f"The following features produced invalid outputs:\n {invalid_features_names[0:20]}\n ... {len(invalid_features_names)} invalid features in total")
        self.call_time = (time.time() - start_time - self.call_time) / (self.n_calls + 1) + self.call_time
        if len(output) != len(self.returns):
            raise ValueError(f"Feature description has different length from feature output ({len(self.returns)} ≠ {len(output)})")
        return output


class MakeFeature:
    """Class decorator to make Feature instances"""
    __slots__ = ['returns']

    def __init__(self, returns):
        self.returns = returns

    def __call__(self, f):
        return Feature(f, self.returns)


# TODO write class for neural network feature, i.e feature that loads and prepares
#  nn feature extraction when it's created


@MakeFeature(
    list(f'outer_hu_moment{i}' for i in range(7)) + list(f'outer_weighted_hu_moment{i}' for i in range(7)) + [
        'outer_eccentricity',
        'outer_solidity',
        'outer_extent',
        'outer_inertia_eigenval0',
        'outer_inertia_eigenval1'
    ] + list(f'inner_hu_moment{i}' for i in range(7)) + list(f'inner_weighted_hu_moment{i}' for i in range(7)) + [
        'inner_eccentricity',
        'inner_solidity',
        'inner_extent',
        'inner_inertia_eigenval0',
        'inner_inertia_eigenval1'
    ])
def region_properties(mask, image):
    r"""Region props, returns 2 regions max (outer and inner), the biggest by area
    """
    if image.shape[2] == 3:  # assume RGB
        image = color.rgb2gray(image)
    all_rp = measure.regionprops(mask.astype(np.int32), intensity_image=image)  # NB: rc coordinates from version >=0.16
    if len(all_rp) == 2:
        outer_rp, inner_rp = all_rp
    elif len(all_rp) > 2:  # this case should not be needed though ?
        outer_rp = max(all_rp, key=lambda rp: rp.area)
        all_rp.remove(outer_rp)
        inner_rp = max(all_rp, key=lambda rp: rp.area)  # second largest by area
    else:
        outer_rp = all_rp.pop()
        inner_rp = None
    outer_features = tuple(outer_rp.moments_hu) + tuple(outer_rp.weighted_moments_hu) + \
                     (outer_rp.eccentricity, outer_rp.solidity, outer_rp.extent) + \
                     tuple(outer_rp.inertia_tensor_eigvals)
    if inner_rp:
        inner_features = tuple(inner_rp.moments_hu) + tuple(inner_rp.weighted_moments_hu) + \
                      (inner_rp.eccentricity, inner_rp.solidity, inner_rp.extent) + \
                      tuple(inner_rp.inertia_tensor_eigvals)
    else:
        inner_features = (0.0,) * len(outer_features)
    return outer_features + inner_features


@MakeFeature(
    list(f'red_haralick_vert{i}' for i in range(13)) + list(f'red_haralick_horz{i}' for i in range(13)) +
    list(f'red_haralick_maj_diag{i}' for i in range(13)) + list(f'red_haralick_min_diag{i}' for i in range(13))
)  # there are 13 stable haralick features per direction (vert, horiz, diagonal 1, diagonal 2)
def red_haralick(image, mask):
    r"""Haralick features for red channel"""
    masked_red_image = np.where(mask[..., np.newaxis].repeat(3, axis=2) > 0, image, 0)[..., 0]  # extract object
    return mh.features.haralick(masked_red_image.astype(np.int32)).flatten()


@MakeFeature(
    list(f'blue_haralick_vert{i}' for i in range(13)) + list(f'blue_haralick_horz{i}' for i in range(13)) +
    list(f'blue_haralick_maj_diag{i}' for i in range(13)) + list(f'blue_haralick_min_diag{i}' for i in range(13))
)  # there are 13 stable haralick features
def blue_haralick(image, mask):
    r"""Haralick features for blue channel"""
    masked_blue_image = np.where(mask[..., np.newaxis].repeat(3, axis=2) > 0, image, 0)[..., 1]  # extract object
    return mh.features.haralick(masked_blue_image.astype(np.int32)).flatten()


@MakeFeature(
    list(f'green_haralick_vert{i}' for i in range(13)) + list(f'green_haralick_horz{i}' for i in range(13)) +
    list(f'green_haralick_maj_diag{i}' for i in range(13)) + list(f'green_haralick_min_diag{i}' for i in range(13))
)  # there are 13 stable haralick features
def green_haralick(image, mask):
    r"""Haralick features for green channel"""
    masked_green_image = np.where(mask[..., np.newaxis].repeat(3, axis=2) > 0, image, 0)[..., 2]  # extract object
    return mh.features.haralick(masked_green_image.astype(np.int32)).flatten()


@MakeFeature(
    list(f'gray_haralick_vert{i}' for i in range(13)) + list(f'gray_haralick_horz{i}' for i in range(13)) +
    list(f'gray_haralick_maj_diag{i}' for i in range(13)) + list(f'gray_haralick_min_diag{i}' for i in range(13))
)  # there are 13 stable haralick features
def gray_haralick(gray_image, mask):
    r"""Haralick features for gray channel"""
    masked_gray_image = np.where(mask > 0, gray_image, 0)
    return mh.features.haralick(masked_gray_image.astype(np.int32)).flatten()


N_SURF_POINTS = 100


@MakeFeature(list(f'surf_p{i}_f{j}' for i in range(N_SURF_POINTS) for j in range(70)))
def surf_points(gray_image):
    # if descriptor_only is false, the location of interest points is also returned
    points = surf.surf(gray_image, nr_octaves=4, nr_scales=6, initial_step_size=1, threshold=0.1,
                       max_points=N_SURF_POINTS)
    # max points smaller than defaults as we look at small images
    if points.shape[0] < N_SURF_POINTS:
        points = np.pad(points, ((0, N_SURF_POINTS - points.shape[0]), (0, 0)), 'constant')
    elif points.shape[0] > N_SURF_POINTS:
        points = points[:N_SURF_POINTS]
    return points.flatten()


distances, angles, num_levels = [5, 20, 100], [0, np.pi/4, np.pi/2, 3*np.pi/4], 8


@MakeFeature(list(f'gray_contrast_d{d}_a{a}' for d in distances for a in angles) +
             list(f'gray_correlation_d{d}_a{a}' for d in distances for a in angles))  # d1a1 d1a2 d1a3 ...
def gray_cooccurrence(gray_image):
    # bin images:
    assert 256 % num_levels == 0, "Must divide number of grayscale levels perfectly"
    binned_image = np.copy(gray_image)
    for i in range(num_levels):
        binned_image[np.logical_and(gray_image >= 256/num_levels*i, gray_image < 256/num_levels*(i + 1))] = i
    p = feature.greycomatrix(binned_image, distances, angles, levels=num_levels)
    p_contrast = feature.greycoprops(p, 'contrast')
    p_correlation = feature.greycoprops(p, 'correlation')
    return list(p_contrast.flatten()) + list(p_correlation.flatten())


NUM_KEYPOINTS = 30
orb_extractor = feature.ORB(n_keypoints=NUM_KEYPOINTS)


@MakeFeature(list(f'orb_d{i}:{j}' for i in range(NUM_KEYPOINTS) for j in range(256)))
def orb_descriptor(gray_image):
    try:
        orb_extractor.detect_and_extract(gray_image)
        descriptors = orb_extractor.descriptors.astype(np.uint8)
        if descriptors.shape[0] < NUM_KEYPOINTS or descriptors.shape[1] < 256:
            descriptors = np.pad(descriptors,
                                 ((0, NUM_KEYPOINTS - descriptors.shape[0]), (0, 256 - descriptors.shape[1])),
                                 'constant')
    except RuntimeError as err:
        if any(arg.startswith('ORB found no features') for arg in err.args):
            descriptors = np.zeros((NUM_KEYPOINTS, 256))
        else:
            raise
    return descriptors.astype(np.uint8).flatten()


# nuclear features  TODO test !!!

rp_names = list(f'outer_hu_moment{i}' for i in range(7)) + list(f'outer_weighted_hu_moment{i}' for i in range(7)) + [
        'outer_eccentricity',
        'outer_solidity',
        'outer_extent',
        'outer_inertia_eigenval0',
        'outer_inertia_eigenval1'
    ]

# TODO test
@MakeFeature(['nuclear_' + rp_name for rp_name in rp_names] + ['nuclear_density'])
def nuclear_features(nuclear_mask, nuclear_image, mask):
    assert nuclear_mask.shape[:2] == mask.shape[:2], "the nuclear mask and image have the same shape"
    nuclear_mask[mask == 0] = 0  # zero out all nuclei that aren't contained within glands
    nuclear_image[mask == 0] = 0
    nuclear_rps = measure.regionprops(nuclear_mask, nuclear_image)
    features = {'moments_hu': [], 'eccentricity': [], 'solidity': [], 'extent': [], 'inertia_tensor_eigenvals': []}
    nuclear_perimeter, previous_nuclear_rp = 0, None  # compute distance connecting all nuclei in an image
    for nuclear_rp in nuclear_rps:
        for feature_name, feature_values in features.items():
            feature_values.append(getattr(nuclear_rp, feature_name))
        if previous_nuclear_rp is not None:
            nuclear_perimeter += np.linalg.norm(nuclear_rp.centroid - previous_nuclear_rp.centroid, ord=2)
        previous_nuclear_rp = nuclear_rp
    mean_features = {}
    for name, values in features.items():
        if isinstance(values[0], Real):
            mean_features[name] = sum(values)/len(values)
        elif isinstance(values[0], (tuple, list)):
            values = np.array(values)
            for index, value in enumerate(values):
                mean_features[f'{name}{index}'] = np.mean(values[:, index])
    nuclear_density = nuclear_perimeter / len(nuclear_rps)
    return tuple(mean_features.values()) + (nuclear_density,)


