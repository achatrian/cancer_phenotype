import sys
import os
sys.path.extend(['/well/rittscher/users/achatrian/cancer_phenotype/base',
                 '/well/rittscher/users/achatrian/cancer_phenotype'])
import multiprocessing as mp
from pathlib import Path
from base.options.train_options import TrainOptions
from base.data.wsi_reader import WSIReader


class SlidePreprocessor(mp.Process):

    def __init__(self, opt, queue, process_id):
        super().__init__(name='SlidePreprocessor')
        self.daemon = True
        self.opt = opt
        self.queue = queue
        self.process_id = process_id

    def run(self):
        while True:
            file = self.queue.get()
            if file is None:
                self.queue.task_done()/mnt/rescomp/projects/TCGA_prostate/TCGA
                break
            slide = WSIReader(self.opt, file)
            print(f"[{self.process_id}] processing {os.path.basename(file)}")
            slide.find_tissue_locations()
            slide.export_tissue_tiles()
            self.queue.task_done()


def main():
    opt = TrainOptions().parse()
    root_path = Path(opt.data_dir)
    paths = root_path.glob('**/*.svs')
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
    # save all into a json file
    WSIReader.save_quality_control_to_json(opt, files)
    print(f"Saved qc data for {len(files)} files to json")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
