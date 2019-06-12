import json
from pathlib import Path
from base.data.base_dataset import BaseDataset
from quant import read_annotations, find_overlap
from base.utils.annotation_builder import AnnotationBuilder
from base.data.wsi_reader import WSIReader


class GlandDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.slides_paths = [path for path in Path(self.opt.data_dir).iterdir() if path.suffix in ('.svs', '.ndpi')]
        self.slides = [WSIReader(opt, path) for path in self.slides_paths]
        self.slide_ids = [Path(slide.file_name).name[:-4] for slide in self.slides]
        self.annotation_paths = list(path for path in (Path(self.opt.data_dir)/'data'/'annotations').iterdir()
                                     if path.suffix == '.json')
        contour_struct = read_annotations(Path(self.opt.data_dir), slide_ids=self.slide_ids)
        unloaded = []
        self.contours = dict()
        for i, slide_id in enumerate(self.slide_ids):
            try:
                # look for
                annotation_path = next(path for path in self.annotation_paths if slide_id in path.name)
            except StopIteration:
                unloaded.append(slide_id)
                continue
            annotation = AnnotationBuilder.from_annotation_path(annotation_path)
            contour_lib = contour_struct[slide_id]
            self.contours[slide_id] = contour_lib

    def name(self):
        return "GlandDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
