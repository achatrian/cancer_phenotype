from pathlib import Path
from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def __init__(self):
        super(TrainOptions, self).__init__()
        parser = self.parser
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--no_visdom', action='store_true', help="Do not use visdom if this flag is given")
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--val_epoch_freq', type=int, default=10, help="epoch interval between rounds of evaluation on the validation set [-1 prevents validation] ")
        parser.add_argument('--val_batch_size', default=20, type=int)
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--nepoch', type=int, default=100, help='# of epochs at starting learning rate')
        parser.add_argument('--nepoch_decay', type=int, default=100, help='# of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--html', action='store_true', help='save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine|cyclic')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--message', type=str, default='', help="Text saved to experiment dir as a file")
        self.parser = parser
        self.is_train = True

    def parse(self):
        super().parse()
        if self.opt.message:
            expr_dir = Path(self.opt.checkpoints_dir) / self.opt.experiment_name
            with (expr_dir/'experiment_readme.txt').open('w') as message_file:
                message_file.write(self.opt.message)
        return self.opt


