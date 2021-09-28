import torch
from torch import nn
from torch import optim
from base.models.base_model import BaseModel
from .networks import Encoders, Dense_Encoders, BiasReduceLoss, TotalVaryLoss, SelfSmoothLoss2
from .networks import DecodersIntegralWarper2 as Decoders
from .networks import Dense_DecodersIntegralWarper2 as Dense_Decoders
from encode.utils import dae_utils as utils


class DeformingAEModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.module_names = ['Encoders', 'Decoders']
        self.netEncoders = Encoders(opt)
        self.netDecoders = Decoders(opt)
        self.loss_names = ['recon', 'tv_warp', 'bias_reduce', 'self_smooth_l1', 'self_smooth_l2', 'total']
        self.metric_names = ['totalm']
        self.recon = nn.L1Loss()
        self.tv_warp = TotalVaryLoss(opt)
        self.bias_reduce = BiasReduceLoss(opt)
        self.self_smooth_l1 = TotalVaryLoss(opt)
        self.self_smooth_l2 = SelfSmoothLoss2(opt)
        self.total = None
        self.visual_names = ['input', 'output']
        self.visual_types = ['image', 'image']
        self.loss_recon, self.loss_tv_warp, self.loss_bias_reduce, self.loss_smooth_l1, self.loss_smooth_l2 = (None,)*5
        self.loss_total = None
        if self.is_train:
            self.optimizers = [
                optim.Adam(self.netEncoders.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999)),
                optim.Adam(self.netDecoders.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
            ]
        self.input, self.mask, self.output, self.image_paths = (None,)*4

    def name(self):
        return "DeformingAEModel"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # TODO complete
        parser.add_argument('--latent_dim', type=int, default=128, help="dimensionality of general latent code (before disentangling)")
        parser.add_argument('--warp_grid_dim', type=int, default=128, help="dimensionality of warping grid (deformation field) latent code")
        parser.add_argument('--texture_dim', type=int, default=16, help="dimensionality of texture latent code")
        parser.add_argument('--num_gen_filters', type=int, default=32, help="number of filters in generator")
        parser.set_defaults(num_filters=32)  # set number of filters in discriminator
        # defaults are for patch size = 64 (Korsuk used default parameters) # TODO scale up?
        return parser
    
    def set_input(self, data):
        # talk of mask rather than target, as target is input itself
        self.input = data['input']  # 1
        self.visual_paths = {'input': data['input_path'],
                             'output': [''] * len(data['input_path'])}  # and 3 must be returned by dataset
        if self.opt.gpu_ids and (self.input.device.type == 'cpu' or self.mask.device.type == 'cpu'):
            self.input = self.input.cuda(device=self.device)

    def forward(self):
        dp0_img = self.input
        baseg = utils.getBaseGrid(N=self.opt.patch_size, getbatch=True, batchSize=dp0_img.size()[0])
        zero_warp = torch.cuda.FloatTensor(1, 2, self.opt.patch_size, self.opt.patch_size).fill_(0)
        if self.opt.gpu_ids:
            dp0_img, baseg, zero_warp = utils.setCuda(dp0_img, baseg, zero_warp)
        baseg, zero_warp = baseg.requires_grad_(False), zero_warp.requires_grad_(False)
        self.netEncoders.zero_grad()
        self.netDecoders.zero_grad()
        dp0_z, dp0_zI, dp0_zW = self.netEncoders(dp0_img)
        dp0_I, dp0_W, dp0_output, dp0_Wact = self.netDecoders(dp0_zI, dp0_zW, baseg)
        self.loss_recon = self.recon(dp0_output, dp0_img)
        self.loss_tv_warp = self.tv_warp(dp0_W, weight=1e-6)  # smooth warping loss
        self.loss_bias_reduce = self.bias_reduce(dp0_W, zero_warp, weight=1e-2)
        self.loss_total = self.loss_recon + self.loss_tv_warp + self.loss_bias_reduce
        self.output = dp0_output

    def backward(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        self.loss_total.backward()
        for optimizer in self.optimizers:
            optimizer.step()

    def optimize_parameters(self):
        self.set_requires_grad(self.netEncoders, requires_grad=True)
        self.set_requires_grad(self.netDecoders, requires_grad=True)
        self.forward()
        self.backward()
        for scheduler in self.schedulers:
            # step for schedulers that update after each iteration
            try:
                scheduler.batch_step()
            except AttributeError:
                pass

    def evaluate_parameters(self):
        self.u_('recon', self.loss_recon)  # reconstruction loss
        self.u_('tv_warp', self.loss_tv_warp)
        self.u_('bias_reduce', self.loss_bias_reduce)  # bias reduce loss
        self.u_('total', self.loss_total)
        self.u_('totalm', self.loss_total)  # there must be at least one metric











