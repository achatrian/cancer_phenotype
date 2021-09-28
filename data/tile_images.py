from pathlib import Path
import argparse
from imageio import imread, imwrite
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--tile_dirname', type=str, default='tiles')
    parser.add_argument('--mask_dir', type=str, default=None)
    args = parser.parse_args()

    image_paths = list((path for path in Path(args.data_dir).iterdir()
                       if path.suffix == '.png' or path.suffix == '.jpg'))
    image_paths += list(path for path in Path(args.data_dir).glob('*/*.png'))
    image_paths += list(path for path in Path(args.data_dir).glob('*/*.jpg'))
    image_paths = sorted(image_paths, key=lambda p: p.name)
    if args.mask_dir is not None:
        mask_paths = [Path(args.data_dir)/'data'/args.mask_dir/path.name for path in image_paths]
        mask_paths += [path.with_suffix('.png') for path in mask_paths]
        mask_paths += [path.with_suffix('.jpg') for path in mask_paths]
        mask_paths = [path for path in mask_paths if path.exists()]
        assert len(image_paths) == len(mask_paths)
    tile_dir = Path(args.data_dir/'data'/args.tile_dirname)
    tile_dir.mkdir(exist_ok=True, parents=True)
    images_dir = tile_dir/'images'
    images_dir.mkdir(exist_ok=True, parents=True)
    masks_dir = tile_dir/'masks'
    masks_dir.mkdir(exist_ok=True, parents=True)
    for i, image_path in enumerate(tqdm(image_paths)):
        image = imread(image_path)
        for j in range(0, image.shape[0], args.tile_size):
            for k in range(0, image.shape[1], args.tile_size):
                tile = image[j:j + args.tile_size, k:k + args.tile_size]
                imwrite(images_dir /f'{image_path.with_suffix("").name}_{j}_{k}.png', tile)
        if args.mask_dir is not None:
            mask = imread(mask_paths[i])
            for j in range(0, mask.shape[0], args.tile_size):
                for k in range(0, mask.shape[1], args.tile_size):
                    mask_tile = mask[j:j + args.tile_size, k:k + args.tile_size]
                    imwrite(masks_dir/f'{image_path.with_suffix("").name}_{j}_{k}.png', mask_tile)
