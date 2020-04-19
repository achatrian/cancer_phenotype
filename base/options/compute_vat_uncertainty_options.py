from .base_options import BaseOptions


class ComputeVATUncertaintyOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument('--slide_id', type=str, help="slide id to process")
        parser.add_argument('--area_annotation_dir', type=str, default='tumour_area_annotations', help="Name of subdir where delimiting area annotations can be found")
        parser.add_argument('--tissue_content_threshold', type=float, default=0.15, help="Tiles with lower tissue content than this value are left blank (filled with 0)")
        parser.add_argument('--distance_threshold', type=float, default=0.01, help="Value multiplied by peak in distance transform of mask to threshold objects")
        parser.add_argument('--shuffle_images', action='store_true', help="Process images in data_dir in a random order")
        # WSIReader options
        parser.add_argument('--qc_mpp', default=2.0, type=float, help="MPP value to perform quality control on slide")
        parser.add_argument('--mpp', default=0.50, type=float, help="MPP value to read images from slide")
        parser.add_argument('--check_tile_blur', action='store_true', help="Check for blur")
        parser.add_argument('--check_tile_fold', action='store_true', help="Check tile fold")
        parser.add_argument('--overwrite_qc', action='store_true', help="Overwrite saved quality control data")

        parser.set_defaults(model='vat_uncertainty')
        self.is_train = False
        self.is_apply = True
        self.parser = parser
