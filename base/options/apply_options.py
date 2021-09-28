from base.options.base_options import BaseOptions


class ApplyOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument('--image_suffix', type=str, default=['png'], action='append', choices=['svs', 'ndpi', 'png', 'jpg'], help="create annotations for images with this extension")
        parser.add_argument('--no_visdom', action='store_true', help="Do not use visdom if this flag is given")
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing applying results on console')
        parser.add_argument('--make_subset', action='store_true', help="Use make_subset method to select only part of dataset")
        parser.add_argument('--extract_contours', action='store_true', help="Extract contours from output segmentation masks and save them in annotation")
        parser.set_defaults(load_size=parser.get_default('fine_size'), eval=True)  # To avoid cropping, the loadSize should be the same as fineSize
        self.is_train = False
        self.is_apply = True
        self.parser = parser
