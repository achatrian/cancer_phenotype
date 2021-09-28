import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


###################################
######### loss functions ##########
###################################

class BiasReduceLoss(nn.Module):
    def __init__(self, opt):
        super(BiasReduceLoss, self).__init__()
        self.opt = opt
        self.criterion = nn.MSELoss()
        self.register_buffer('w', torch.tensor(1, requires_grad=False))
    def forward(self, x, y, weight=1):
        self.w.fill_(weight)
        self.avg = torch.mean(x, 0).unsqueeze(0)
        self.loss = self.w*self.criterion(self.avg, y)
        return self.loss

class TotalVaryLoss(nn.Module):
    def __init__(self,opt):
        super(TotalVaryLoss, self).__init__()
        self.opt = opt
        self.register_buffer('w', torch.tensor(1, requires_grad=False))

    def forward(self, x, weight=1):
        self.w.fill_(weight)
        self.loss = self.w * (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        return self.loss

class SelfSmoothLoss2(nn.Module):
    def __init__(self,opt):
        super(SelfSmoothLoss2, self).__init__()
        self.opt = opt
        self.register_buffer('w', torch.tensor(1, requires_grad=False))
    def forward(self, x, weight=1):
        self.w.fill_(weight)
        self.x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
        self.y_diff = x[:, :, :-1, :] - x[:, :, 1:, :]
        self.loss = torch.sum(torch.mul(self.x_diff, self.x_diff)) + torch.sum(torch.mul(self.y_diff, self.y_diff))
        self.loss = w * self.loss
        return self.loss


###################################
#########  basic blocks  ##########
###################################
# a mixer (linear layer)
class waspMixer(nn.Module):
    def __init__(self, nin=128, nout=128):
        super(waspMixer, self).__init__()
        self.main = nn.Sequential(
            # simply a linear layer
            nn.Linear(nin, nout),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output

# shading * albedo = texture(img)
class waspIntrinsicComposer(nn.Module):
    def __init__(self, opt):
        super(waspIntrinsicComposer, self).__init__()
        self.ngpu = opt.ngpu
        self.input_channels = opt.input_channels
    def forward(self, shading, albedo):
        self.shading = shading.repeat(1,self.input_channels,1,1)
        self.img = torch.mul(self.shading, albedo)
        return self.img

# warp images according to the grid
class waspWarper(nn.Module):
    def __init__(self, opt):
        super(waspWarper, self).__init__()
        self.opt = opt
        self.batchSize = opt.batch_size
        self.imgSize = opt.patch_size

    def forward(self, input_img, input_grid):
        self.warp = input_grid.permute(0,2,3,1)
        self.output = F.grid_sample(input_img, self.warp)
        return self.output

# integrate over the predicted grid offset to get the grid(deformation field)
class waspGridSpatialIntegral(nn.Module):
    def __init__(self, opt):
        super(waspGridSpatialIntegral, self).__init__()
        self.opt = opt
        self.w = self.opt.patch_size
        filterx = torch.tensor(torch.ones(1, 1, 1, self.w), dtype=torch.float, requires_grad=False)
        self.register_buffer('filterx', filterx)
        filtery = torch.tensor(torch.ones(1, 1, self.w, 1), dtype=torch.float, requires_grad=False)
        self.register_buffer('filtery', filtery)
    def forward(self, input_diffgrid):
        fullx = F.conv_transpose2d(input_diffgrid[:, 0, :, :].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:, 1, :, :].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:, :, 0:self.w, 0:self.w], fully[:, :, 0:self.w, 0:self.w]),1)
        return output_grid

# an encoder architecture
class waspEncoder(nn.Module):
    def __init__(self, input_channels=1, num_filters=32, num_dim=128):
        super(waspEncoder, self).__init__()
        self.num_dim = num_dim
        self.main = nn.Sequential(
            # input is (input_channels) x 64 x 64
            nn.Conv2d(input_channels, num_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (num_filters) x 32 x 32
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (num_filters*2) x 16 x 16
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (num_filters*4) x 8 x 8
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (num_filters*8) x 4 x 4
            nn.Conv2d(num_filters * 8, num_dim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1, self.num_dim)
        #print(output.size())
        return output

# a decoder architecture
class waspDecoder(nn.Module):
    def __init__(self, nz=128, input_channels=1, num_gen_filters=32, lb=0, ub=1):
        # ngf --> num_gen_filters
        super(waspDecoder, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, num_gen_filters * 8, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(num_gen_filters * 8),
            nn.ReLU(True),
            # state size. (num_gen_filters*8) x 4 x 4
            nn.ConvTranspose2d(num_gen_filters * 8, num_gen_filters * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_gen_filters * 4),
            nn.ReLU(True),
            # state size. (num_gen_filters*4) x 8 x 8
            nn.ConvTranspose2d(num_gen_filters * 4, num_gen_filters * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_gen_filters * 2),
            nn.ReLU(True),
            # state size. (num_gen_filters*2) x 16 x 16
            nn.ConvTranspose2d(num_gen_filters * 2,num_gen_filters, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_gen_filters),
            nn.ReLU(True),
            # state size. (num_gen_filters) x 32 x 32
            nn.ConvTranspose2d(num_gen_filters, num_gen_filters, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_gen_filters),
            nn.ReLU(True),
            # state size. (input_channels) x 64 x 64
            nn.ConvTranspose2d(num_gen_filters, input_channels, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        output = self.main(input)
        return output

# a decoder architecture
class waspDecoderTanh(nn.Module):
    def __init__(self, nz=128, input_channels=1, num_gen_filters=32, lb=0, ub=1):
        super(waspDecoderTanh, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, num_gen_filters * 8, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(num_gen_filters * 8),
            nn.Tanh(),
            # state size. (num_gen_filters*8) x 4 x 4
            nn.ConvTranspose2d(num_gen_filters * 8, num_gen_filters * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_gen_filters * 4),
            nn.Tanh(),
            # state size. (num_gen_filters*4) x 8 x 8
            nn.ConvTranspose2d(num_gen_filters * 4, num_gen_filters * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_gen_filters * 2),
            nn.Tanh(),
            # state size. (num_gen_filters*2) x 16 x 16
            nn.ConvTranspose2d(num_gen_filters * 2,num_gen_filters, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_gen_filters),
            nn.Tanh(),
            # state size. (num_gen_filters) x 32 x 32
            nn.ConvTranspose2d(num_gen_filters, num_gen_filters, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(num_gen_filters),
            nn.Tanh(),
            # state size. (input_channels) x 64 x 64
            nn.ConvTranspose2d(num_gen_filters, input_channels, 3, 1, 1, bias=False),
            #nn.Hardtanh(lb,ub),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output

###################################
########  densenet blocks #########
###################################

class DenseBlockEncoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockEncoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers     = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.InstanceNorm2d(n_channels),
                    activation(*args),
                    nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseBlockDecoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockDecoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.InstanceNorm2d(n_channels),
                    activation(*args),
                    nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseTransitionBlockEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp             = mp
        self.main           = nn.Sequential(
                nn.InstanceNorm2d(n_channels_in),
                activation(*args),
                nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(mp),
        )
    def forward(self, inputs):
        return self.main(inputs)


class DenseTransitionBlockDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.main           = nn.Sequential(
                nn.InstanceNorm2d(n_channels_in),
                activation(*args),
                nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False),
        )
    def forward(self, inputs):
        return self.main(inputs)

class waspDenseEncoder(nn.Module):
    def __init__(self, opt, ngpu=1, input_channels=1, num_filters=32, num_dim = 128, activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Sigmoid, f_args=[]):
        super(waspDenseEncoder, self).__init__()
        self.ngpu = ngpu
        self.num_dim = num_dim

        self.main = nn.Sequential(
                # input is (input_channels) x 64 x 64
                nn.Conv2d(input_channels, num_filters, 4, stride=2, padding=1),

                # state size. (num_filters) x 32 x 32
                DenseBlockEncoder(num_filters, 6),
                DenseTransitionBlockEncoder(num_filters, num_filters*2, 2, activation=activation, args=args),

                # state size. (num_filters*2) x 16 x 16
                DenseBlockEncoder(num_filters*2, 12),
                DenseTransitionBlockEncoder(num_filters*2, num_filters*4, 2, activation=activation, args=args),

                # state size. (num_filters*4) x 8 x 8
                DenseBlockEncoder(num_filters*4, 24),
                DenseTransitionBlockEncoder(num_filters*4, num_filters*8, 2, activation=activation, args=args),

                # state size. (num_filters*8) x 4 x 4
                DenseBlockEncoder(num_filters*8, 16),
                DenseTransitionBlockEncoder(num_filters*8, num_dim, 4, activation=activation, args=args),
                f_activation(*f_args),
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.num_dim)
        return output

class waspDenseDecoder(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128, input_channels=1, num_gen_filters=32, lb=0, ub=1, activation=nn.ReLU, args=[False], f_activation=nn.Hardtanh, f_args=[0,1]):
        super(waspDenseDecoder, self).__init__()
        self.ngpu   = ngpu
        self.main   = nn.Sequential(
            # input is Z, going into convolution
            nn.ConvTranspose2d(nz, num_gen_filters * 8, 4, 1, 0, bias=False),

            # state size. (num_gen_filters*8) x 4 x 4
            DenseBlockDecoder(num_gen_filters*8, 16),
            DenseTransitionBlockDecoder(num_gen_filters*8, num_gen_filters*4),

            # state size. (num_gen_filters*4) x 8 x 8
            DenseBlockDecoder(num_gen_filters*4, 24),
            DenseTransitionBlockDecoder(num_gen_filters*4, num_gen_filters*2),

            # state size. (num_gen_filters*2) x 16 x 16
            DenseBlockDecoder(num_gen_filters*2, 12),
            DenseTransitionBlockDecoder(num_gen_filters*2, num_gen_filters),

            # state size. (num_gen_filters) x 32 x 32
            DenseBlockDecoder(num_gen_filters, 6),
            DenseTransitionBlockDecoder(num_gen_filters, num_gen_filters),

            # state size (num_gen_filters) x 64 x 64
            nn.InstanceNorm2d(num_gen_filters),
            activation(*args),
            nn.ConvTranspose2d(num_gen_filters, input_channels, 3, stride=1, padding=1, bias=False),
            f_activation(*f_args),
        )
    def forward(self, inputs):
        return self.main(inputs)


###################################
###### encoders and decoders ######
###################################

#### The encoders ####

# encoders of DAE
class Encoders(nn.Module):
    def __init__(self, opt):
        # zdim --> latent_dim; wdim --> warp_grid_dim; idim --> texture_dim
        super(Encoders, self).__init__()
        self.encoder = waspEncoder(input_channels=opt.input_channels, num_filters=opt.num_filters,
                                   num_dim=opt.latent_dim)
        self.zImixer = waspMixer(nin=opt.latent_dim, nout=opt.texture_dim)
        self.zWmixer = waspMixer(nin=opt.latent_dim, nout=opt.warp_grid_dim)
        self.z, self.zImg, self.zWarp = (None,) * 3

    def forward(self, input_):
        self.z = self.encoder(input_)
        self.zImg = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp


# encoders of instrinsic DAE
class Encoders_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Encoders_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(input_channels=opt.input_channels, num_filters=opt.num_filters,
                                   num_dim=opt.latent_dim)
        #self.zImixer = waspMixer(opt, ngpu=1, nin = opt.latent_dim, nout = opt.texture_dim)
        self.zSmixer = waspMixer(nin=opt.latent_dim, nout=opt.sdim)
        self.zTmixer = waspMixer(nin=opt.latent_dim, nout=opt.tdim)
        self.zWmixer = waspMixer(nin=opt.latent_dim, nout=opt.warp_grid_dim)

    def forward(self, input):
        self.z     = self.encoder(input)
        #self.zImg  = self.zImixer(self.z)
        self.zShade = self.zSmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp


# encoders of DAE, using DenseNet architecture
class Dense_Encoders(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, input_channels=opt.input_channels, num_filters = opt.num_filters, num_dim = opt.latent_dim)
        self.zImixer = waspMixer(nin=opt.latent_dim, nout=opt.texture_dim)
        self.zWmixer = waspMixer(nin=opt.latent_dim, nout=opt.warp_grid_dim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp

# encoders of Intrinsic DAE, using DenseNet architecture
class Dense_Encoders_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, input_channels=opt.input_channels, num_filters = opt.num_filters, num_dim = opt.latent_dim)
        self.zSmixer = waspMixer(nin=opt.latent_dim, nout=opt.sdim)
        self.zTmixer = waspMixer(nin=opt.latent_dim, nout=opt.tdim)
        self.zWmixer = waspMixer(nin=opt.latent_dim, nout=opt.warp_grid_dim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zShade = self.zSmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp


#### The decoders ####

# decoders of DAE
class DecodersIntegralWarper2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2, self).__init__()
        self.imagedimension = opt.patch_size
        self.texture_dim = opt.texture_dim
        self.warp_grid_dim = opt.warp_grid_dim
        self.decoderI = waspDecoder(nz=opt.texture_dim, input_channels=opt.input_channels,
                                    num_gen_filters=opt.num_gen_filters, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(nz=opt.warp_grid_dim, input_channels=2, num_gen_filters=opt.num_gen_filters,
                                        lb=0, ub=0.1)
        self.warper   = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)

    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.texture_dim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.warp_grid_dim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid.float()
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping


# decoders of intrinsic DAE
class DecodersIntegralWarper2_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2_Intrinsic, self).__init__()
        self.imagedimension = opt.patch_size
        self.ngpu = opt.ngpu
        self.texture_dim = opt.texture_dim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.warp_grid_dim = opt.warp_grid_dim
        self.decoderS = waspDecoder(nz=opt.sdim, input_channels=1, num_gen_filters=opt.num_gen_filters, lb=0, ub=1)
        self.decoderT = waspDecoder(nz=opt.tdim, input_channels=opt.input_channels, num_gen_filters=opt.num_gen_filters,
                                    lb=0, ub=1)
        self.decoderW = waspDecoderTanh(nz=opt.warp_grid_dim, input_channels=2, num_gen_filters=opt.num_gen_filters,
                                        lb=0, ub=0.1)
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        self.warper   = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1, 1)

    def forward(self, zS, zT, zW, basegrid):
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1,self.warp_grid_dim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping


# decoders of DAE, using DenseNet architecture
class Dense_DecodersIntegralWarper2(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2, self).__init__()
        self.imagedimension = opt.patch_size
        self.ngpu = opt.ngpu
        self.texture_dim = opt.texture_dim
        self.warp_grid_dim = opt.warp_grid_dim
        self.decoderI = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.texture_dim, input_channels=opt.input_channels, num_gen_filters=opt.num_gen_filters, lb=0, ub=1)
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.warp_grid_dim, input_channels=2, num_gen_filters=opt.num_gen_filters, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.warper = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.img = self.decoderI(zI.view(-1,self.texture_dim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.warp_grid_dim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.img, self.resWarping, self.output, self.warping

# decoders of Intrinsic DAE, using DenseNet architecture
class Dense_DecodersIntegralWarper2_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2_Intrinsic, self).__init__()
        self.imagedimension = opt.patch_size
        self.ngpu = opt.ngpu
        self.texture_dim = opt.texture_dim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.warp_grid_dim = opt.warp_grid_dim
        # shading decoder
        self.decoderS = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, input_channels=1, num_gen_filters=opt.num_gen_filters, lb=0, ub=1)
        # albedo decoder
        self.decoderT = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, input_channels=opt.input_channels, num_gen_filters=opt.num_gen_filters, lb=0, ub=1)
        # deformation decoder
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.warp_grid_dim, input_channels=2, num_gen_filters=opt.num_gen_filters, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        # shading*albedo=texture
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        # deformation offset decoder
        self.warper   = waspWarper(opt)
        # spatial intergrator for deformation field
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zS, zT, zW, basegrid):
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img     = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1,self.warp_grid_dim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping
