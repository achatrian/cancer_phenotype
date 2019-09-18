from pytest import fixture
import imageio
import numpy as np
from annotation.mask_converter import MaskConverter
from quant import contour_to_mask


@fixture(params=[1, 2])
def iters(request):
    return request.param


@fixture
def mask(iters):
    return imageio.imread(f'/home/andrea/Documents/Temp/Data/test_mask{iters}.png')


def test_mask_to_contour(mask, iters):
    converter = MaskConverter(fix_ambiguity=True)
    contours, labels, boxes = converter.mask_to_contour(mask)
    painted_mask = np.zeros_like(mask)
    print(painted_mask.shape)
    for contour, label in zip(contours, labels):
        painted_mask = contour_to_mask(contour, value=200 if label == 'epithelium' else 250, mask=painted_mask,
                                       mask_origin=(0, 0))
    imageio.imwrite(f'/home/andrea/Documents/Temp/Data/converted_mask{iters}.png', painted_mask)
