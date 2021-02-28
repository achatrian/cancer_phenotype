import argparse
from pathlib import Path
import multiprocessing as mp
import imageio
from tqdm import tqdm
import pandas as pd
from data.images import is_hne
from base.utils import debug


r"""Delete all tiles with """
"""THIS ISN'T NEEDED IF YOU DON'T GIVE A HUGE MINIMUM SIZE"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--tissue_threshold', type=float, default=0.1)
    parser.add_argument('--ihc_data_file', type=Path, default='/well/rittscher/projects/IHC_Request/data/documents/additional_data_2020-04-21.csv')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    tiles_dir = args.data_dir/'data'/'tiles'
    discarded_tiles_dir = args.data_dir/'data'/f'discarded_tiles_th{args.tissue_threshold}'
    discarded_tiles_dir.mkdir(exist_ok=True, parents=True)
    slide_data = pd.read_csv(args.ihc_data_file, index_col=0)
    slide_data = slide_data.set_index('Image')

    def remove_empty_tiles(slide_path):
        try:  # Check slide is H&E: H&E cases are read in as nan
            if isinstance(slide_data.loc[slide_path.name]['Staining code'].iloc[0], str):
                return 0
        except AttributeError:
            if isinstance(slide_data.loc[slide_path.name]['Staining code'], str):
                return 0
        except KeyError:
            print(f'{slide_path.name} not in data file')
            return 0
        num_removed_tiles = 0
        for image_path in slide_path.iterdir():
            if not image_path.name.endswith('image.png'):
                continue
            image = imageio.imread(image_path)
            if not is_hne(image, args.tissue_threshold):
                num_removed_tiles += 1
                image_path.unlink()
                mask_path = image_path.parent / image_path.name.replace('image', 'mask')
                mask = imageio.imread(mask_path)
                mask_path.unlink()
                imageio.imwrite(discarded_tiles_dir / image_path.name, image)
                imageio.imwrite(discarded_tiles_dir / mask_path.name, mask)
        return num_removed_tiles

    # build a per slide per focus break down (each row)
    all_data = []
    focus_paths = sorted(list(tiles_dir.iterdir()), key=lambda p: p.name)
    num_removed_tiles = 0
    if args.workers > 0:
        with mp.Pool() as pool:
            for focus_path in tqdm(focus_paths, desc="Focus num"):
                if not focus_path.is_dir():
                    continue
                slide_paths = list(focus_path.iterdir())
                pbar = tqdm(total=len(slide_paths), desc="slides")

                def update(*a):
                    pbar.update()
                result = pool.map_async(remove_empty_tiles, slide_paths, callback=update)
                num_removed_tiles_per_slide = result.get()
    else:
        for focus_path in tqdm(focus_paths, desc="Focus num"):
            if not focus_path.is_dir():
                continue
            slide_paths = list(focus_path.iterdir())
            num_removed_tiles_per_slide = [remove_empty_tiles(slide_path)
                                           for slide_path in tqdm(total=len(slide_paths), desc="slides")]
    print(f"Done! Removed {sum(num_removed_tiles_per_slide)} images")  # FIXME number results as 0 when it isn't












