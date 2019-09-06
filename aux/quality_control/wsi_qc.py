import sys
import os
import argparse
sys.path.extend(['/well/rittscher/users/achatrian/cancer_phenotype/base',
                 '/well/rittscher/users/achatrian/cancer_phenotype'])
from itertools import chain
import multiprocessing as mp
from pathlib import Path
from data.images.wsi_reader import WSIReader


r"Script to perform quality control on all the slides in a directory"


class SlidePreprocessor(mp.Process):

    def __init__(self, opt, queue, process_id, export_tiles=False):
        super().__init__(name='SlidePreprocessor')
        self.daemon = True
        self.opt = opt
        self.queue = queue
        self.process_id = process_id
        self.export_tiles = export_tiles

    def run(self):
        while True:
            file = self.queue.get()
            if file is None:
                self.queue.task_done()
                break
            slide = WSIReader(file, self.opt)
            print(f"[{self.process_id}] processing {os.path.basename(file)}")
            slide.find_tissue_locations(self.opt.tissue_threshold, self.opt.saturation_threshold)
            if self.export_tiles:
                slide.export_tissue_tiles()
            self.queue.task_done()


def main():
    opt = WSIReader.get_reader_options(include_path=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    opt2, extra = parser.parse_known_args()
    setattr(opt, 'data_dir', opt2.data_dir)
    setattr(opt, 'workers',  opt2.workers)
    setattr(opt, 'overwrite_qc', True)
    root_path = Path(opt.data_dir)
    paths = chain(root_path.glob('**/*.svs'), root_path.glob('**/*.ndpi'))
    files = sorted((str(path) for path in paths), key=lambda file: os.path.basename(file))  # sorted by basename
    queue = mp.JoinableQueue(opt.workers*2)
    processes = []
    # quality control of each patch in each slide
    for i in range(opt.workers):
        processes.append(
            SlidePreprocessor(opt, queue, i).start()
        )
    for file in files:
        if Path(file).is_file():
            queue.put(file)
    for j in range(opt.workers):
        queue.put(None)
    queue.join()
    print(f"Saved qc data for {len(files)} files")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
