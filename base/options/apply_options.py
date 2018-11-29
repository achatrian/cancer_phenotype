from .base_options import BaseOptions


class ApplyOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument('--deployer_name', type=str, default='tcga', choices=['tcga'], help="Format of images onto which model is applied")
        parser.add_argument('--save_dir', type=str, default='/well/rittscher/users/achatrian/Results', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--ndeploy_workers', type=int, default=0, help='num of processes used in deployment')
        parser.add_argument('--metadata_dir', type=str, default="/well/rittscher/projects/TCGA_prostate/TCGA/data/datacopy", help='Folder where spreadsheets containing metadata are')
        parser.add_argument('--sample_id_name', type=str, default='case_submitter_id', help='Slide specific identifier that is used to organize the metadata entries - must be 1 per slide')
        parser.set_defaults(load_size=parser.get_default('fine_size'))  # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(deployment='y')
        parser.set_defaults(dataset_name='wsi')
        self.is_train = False
        self.parser = parser
