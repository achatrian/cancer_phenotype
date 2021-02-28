from base.datasets.base_dataset import BaseDataset
from torchvision import datasets, transforms


class MNISTDataset(BaseDataset):

    def __init__(self, opt):
        r"""Wrapper around MNIST dataset - Try out algorithms with this dataset"""
        super().__init__(opt)
        self.mnist = datasets.MNIST(self.opt.data_dir, train=(self.opt.phase == 'train'), download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize((self.opt.patch_size,) * 2, interpolation=0),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(patch_size=299)
        return parser

    def __getitem__(self, i):
        image, target = self.mnist[i]
        image = image.repeat(3, 1, 1)  # 3 channels for inception
        return {'input': image, 'target': target, 'input_path': '', 'target_path': ''}

    def __len__(self):
        return len(self.mnist)

