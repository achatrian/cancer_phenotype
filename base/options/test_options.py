from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        parser = self.parser
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--slide_id', type=str, default='', help="If given, it restricts processing to only one slide")

        parser.set_defaults(phase='test', patch_size=parser.get_default('fine_size'))
        parser.set_defaults()
        self.is_train = False
        self.parser = parser


