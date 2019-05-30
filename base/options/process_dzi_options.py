from .base_options import BaseOptions


class ProcessDZIOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument('--slide_id', type=str, help="slide id to process")
        parser.add_argument('--mpp',  type=float, default=0.5, help="Target mpp to read slide at")
        parser.add_argument('--area_annotation_dir', type=str, default='tumour_area_annotations', help="Name of subdir where delimiting area annotations can be found")
        parser.add_argument('--distance_threshold', type=float, default=0.1, help="Value multiplied by peak in distance transform of mask to threshold objects")
        self.is_train = False
        self.is_apply = True
        self.parser = parser
