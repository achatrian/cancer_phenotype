from .base_options import BaseOptions


class SalienceOptions(BaseOptions):
    def __init__(self):
        super(SalienceOptions, self).__init__()
        parser = self.parser
        parser.add_argument('--make_subset', action='store_true', help="Use make_subset method to select only part of dataset")
        parser.add_argument('--slide_id', type=str, default='', help="If given, it restricts processing to only one slide")
        parser.add_argument('--n_salience_maps', type=int, default=3, help="Number of salience maps to acquire per dataset subdivision")
        parser.add_argument('--target_network', type=str, default='net', help="Name of network to optimize for input saliency")
        parser.set_defaults(phase='test', patch_size=parser.get_default('fine_size'))
        parser.set_defaults()
        self.is_train = False
        self.parser = parser


