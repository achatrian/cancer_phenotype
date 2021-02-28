from pathlib import Path
import argparse
from PIL import Image
from colorsys import hls_to_rgb
import numpy as np
from annotation.annotation_builder import AnnotationBuilder
from data.images.wsi_reader import WSIReader


red = hls_to_rgb(1, 0.72, 0)
blue = hls_to_rgb(1, 0.72, 180)
max_side_len = 2**14


# TODO test - made originally for making high res picture for IHC paper, but did not use in the end


def paint_tiles(image, points, color):
    color_mask = (np.ones_like(thumbnail) * color * 255).astype(np.uint8)
    for x, y, w, h in points:
        xs, ys, ws, hs = x*scale_factor, y*scale_factor, w*scale_factor, h*scale_factor
        blend = np.array(Image.blend(
            Image.fromarray(image),
            Image.fromarray(color_mask),
            alpha=1.0))  # blend tissue images with color mask
        thumbnail[ys:ys+hs, xs:xs+ws] = blend
    return thumbnail


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('slide_path', type=Path)
    parser.add_argument('annotation_path', type=Path)
    args = parser.parse_args()

    slide = WSIReader(args.slide_path)
    annotation = AnnotationBuilder.from_annotation_path(args.annotation_path)
    dim0, dim1 = slide.level_dimensions[0]
    if dim0 >= dim1 >= max_side_len:
        scale_factor = max_side_len/dim0
        side_len1 = scale_factor*dim1
        thumbnail = slide.get_thumbnail((max_side_len, side_len1))
    elif dim1 > dim0 >= max_side_len:
        scale_factor = max_side_len/dim1
        side_len0 = scale_factor*dim0
        thumbnail = slide.get_thumbnail((side_len0, max_side_len))
    else:  # max_side_len >= dim0 and max_side_len >= dim1:
        scale_factor = 1.0
        thumbnail = slide.get_thumbnail((dim0, dim1))
    points_ambiguous = annotation.get_layer_points('Ambiguous')
    points_certain = annotation.get_layer_points('Certain')
    paint_tiles(thumbnail, points_ambiguous, blue)
    paint_tiles(thumbnail, points_certain, red)
    thumbnail = Image.fromarray(thumbnail)
    with open(f'~/Desktop/{args.slide_path.name}.png') as image_file:
        thumbnail.save(image_file, dpi=(400, 400))


