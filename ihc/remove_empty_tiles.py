import argparse
from pathlib import Path
import imageio
from tqdm import tqdm
import pandas as pd
from data.images import is_hne
from base.utils import debug


r"""Delete all tiles with """
"""THIS ISN'T NEEDED IF YOU DON@T GIVE A HUGE MINIMUM SIZE"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--tissue_threshold', type=float, default=0.1)
    parser.add_argument('--ihc_data_file', type=Path, default='/well/rittscher/projects/IHC_Request/data/documents/additional_data_12_12_19.csv')
    args = parser.parse_args()
    tiles_dir = args.data_dir/'data'/'tiles'
    slide_data = pd.read_csv(args.ihc_data_file)
    slide_data = slide_data.set_index('Image')
    # build a per slide per focus break down (each row)
    all_data = []
    focus_paths = sorted(list(tiles_dir.iterdir()), key=lambda p: p.name)
    i = 0
    for focus_path in tqdm(focus_paths, desc="Focus num"):
        if not focus_path.is_dir():
            continue
        i = int(focus_path.name[-1])
        for slide_path in focus_path.iterdir():
            try:
                if slide_data.loc[slide_path.name]['Staining code'].iloc[0]:
                    continue
            except AttributeError:
                if slide_data.loc[slide_path.name]['Staining code']:
                    continue
            except KeyError:
                print(f'{slide_path.name} not in data file')
                continue
            for image_path in slide_path.iterdir():
                if not image_path.name.endswith('image.png'):
                    continue
                image = imageio.imread(image_path)
                if not is_hne(image, args.tissue_threshold):
                    i += 1
                    # if i % 50 == 0:
                    #     debug.show_image(image)  # will also remove ck5 images!
                    image_path.unlink()
                    mask_path = image_path.parent/image_path.name.replace('image', 'mask')
                    mask_path.unlink()












