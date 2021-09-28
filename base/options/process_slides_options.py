from .base_options import BaseOptions


class ProcessSlidesOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument('--slide_id', type=str, action='append', help='only process slides with specified ids. Useful for debugging and dividing tasks')
        parser.add_argument('--image_suffix', type=str, default=['tiff'], action='append', choices=['tiff', 'svs', 'ndpi', 'isyntax'], help="process images with these extension")
        parser.add_argument('--area_annotation_dir', type=str, default='tumour_area_annotations', help="Name of subdir where delimiting area annotations can be found")
        parser.add_argument('--tissue_content_threshold', type=float, default=0.15, help="Tiles with lower tissue content than this value are left blank (filled with 0)")
        parser.add_argument('--distance_threshold', type=float, default=0.01, help="Value multiplied by peak in distance transform of mask to threshold objects")
        parser.add_argument('--mask_dirname', type=str, default='masks', help="Name of directory where masks will be saved")
        parser.add_argument('--shuffle_images', action='store_true', help="Process images in data_dir in a random order")
        parser.add_argument('--no_recursive_search', action='store_true')
        parser.add_argument('--require_tissue_mask', action='store_true', help="Throw error if tissue masks are not available for selected dataset")
        parser.add_argument('--extract_contours', action='store_true', help="After making each segmentation mask, extract contours from it")
        parser.add_argument('--force_base_level_read', action='store_true', help="Regardless of WSI zoom structure, tiles are read from base level and resized while processing (more computationally intensive) -- added because of quality difference between base level and upper levels")
        parser.add_argument('--tissue_mask_dirname', type=str, default='masks')
        parser.add_argument('--save_dir', type=str, default=None, help="Alternative full save dir for processed slides")
        parser.add_argument('--recursive_search', action='store_true', help="Find images to process recursively")
        parser.add_argument('--skip_images', type=str, action='append', default=[], help='skip these slides')
        parser.add_argument('--overwrite', action='store_true')
        # WSIReader options
        parser.add_argument('--tissue_threshold', default=0.01, type=float, help="Minimum tile content of tissue to avoid being rejected")
        parser.add_argument('--saturation_threshold', type=int, default=20,
                            help="Saturation difference threshold of tile for it to be considered a tissue tile")
        parser.add_argument('--qc_mpp', default=2.0, type=float, help="MPP value to perform quality control on slide")
        parser.add_argument('--mpp', default=0.50, type=float, help="MPP value to read images from slide")
        parser.add_argument('--set_mpp', default=None, type=float, help="Slide mpp for slides with unspecified resolution")
        parser.add_argument('--check_tile_blur', action='store_true', help="Check for blur")
        parser.add_argument('--check_tile_fold', action='store_true', help="Check tile fold")
        parser.add_argument('--overwrite_qc', action='store_true', help="Overwrite saved quality control data")
        parser.set_defaults(dataset_name='none')
        self.is_train = False
        self.is_apply = True
        self.parser = parser
