from .base_options import BaseOptions


class ApplyOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument('--deployer_name', type=str, default='tileseg', help="Format of images onto which model is applied")
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--ndeploy_workers', type=int, default=1, help='num of processes used in deployment')
        parser.add_argument('--metadata_dir', type=str, default="/well/rittscher/projects/TCGA_prostate/TCGA/data/datacopy", help='Folder where spreadsheets containing metadata are')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.set_defaults(load_size=parser.get_default('fine_size'))  # To avoid cropping, the loadSize should be the same as fineSize
        self.is_train = False
        self.is_apply = True
        self.parser = parser
