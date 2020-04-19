from segment.models.unet_model import UNetModel
from base.utils.vat_uncertainty import VATUncertainty


class VATUncertaintyModel(UNetModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names += ['vatu']
        self.vatu = VATUncertainty(self.net, ip=self.opt.power_iterations)
        self.vat_mu, self.vat_sigma = None, None

    def forward(self):
        self.vat_mu, self.vat_sigma = self.vatu(self.input)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = UNetModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--power_iterations', type=int, default=1)
        return parser

    def name(self):
        return "VATUncertaintyModel"


