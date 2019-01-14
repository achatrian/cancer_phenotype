import sys
import os
sys.path.extend(['/well/rittscher/users/achatrian/cancer_phenotype/base',
                 '/well/rittscher/users/achatrian/cancer_phenotype'])
import multiprocessing as mp
from pathlib import Path
from base.options.train_options import TrainOptions
from base.data.wsi_reader import WSIReader


def slide_preprocessing(file):
    slide = WSIReader(opt, file)
    slide.find_good_locations()  # TODO this still doesn't work perfectly
    return slide


if __name__ == 'main':
    mp.set_start_method('spawn')
    opt = TrainOptions().parse()
    root_path = Path(opt.data_dir)
    paths = root_path.glob('**/*.svs')
    files = sorted((str(path) for path in paths), key=lambda file: os.path.basename(file))  # sorted by basename
    pool = mp.Pool(opt.workers)
    pool.map(slide_preprocessing, files)


