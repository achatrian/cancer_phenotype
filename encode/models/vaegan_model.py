import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import L1Loss, BCELoss
from base.models.base_model import BaseModel


class VAEModel(BaseModel):

    def __init__(self, opt):
        super(VAEModel, self).__init__(opt)
        self.module_names = ['self.netvae', 'self.netdiscr']
        self.loss_names = ['bce', 'l1']
        self.netvae = self.netvae(opt.patch_size, opt.gpu_ids, opt.num_filt_gen, opt.num_lat_dim)
        self.netdiscr = Discriminator(opt.patch_size, opt.gpu_ids, opt.num_filt_discr)
        self.l1 = L1Loss()  # TODO no reduction for gt weighting
        self.bce = BCELoss()
        opt_dis = torch.optim.Adam(self.netdiscr.parameters(), lr=opt.learning_rate)
        opt_enc = torch.optim.Adam(self.netvae.encoder.parameters(), lr=opt.learning_rate)
        opt_dec = torch.optim.Adam(self.netvae.decoder.parameters(), lr=opt.learning_rate)
        opt_vae = torch.optim.Adam(self.netvae.parameters(), lr=opt.learning_rate)
        self.optimizers = {'opt_dis': opt_dis, 'opt_enc': opt_enc, 'opt_dec': opt_dec, 'opt_vae': opt_vae}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_filt_gen', type=int, default=32)
        parser.add_argument('--num_lat_dim', type=int, default=32)
        parser.add_argument('--num_filt_discr', type=int, default=32)

    def forward(self):
        equilibrium = 0.5
        margin = 0.3
        self.netvae.zero_grad()
        self.encoded, self.reconstructed, self.generated = self.netvae(self.input)  # self.generated does not depend on inputs
        mu = self.encoded[0]
        logvar = self.encoded[1]  # !!! log of sigma^2
        # Prior loss (KL div between recognition model and prior)#######
        kld_elements = (mu.pow(2) + logvar.exp() - 1 - logvar) / 2  # log of variance learned by network (why??)
        self.loss_kld = torch.clamp(torch.mean(kld_elements), max=1e15)

        # self.l1 loss between discriminator layers for self.reconstructed and real input
        self.rec0_loss, self.rec1_loss = self.netdiscr(self.reconstructed, output_layer=True)
        l_real0, l_real1 = self.netdiscr(self.input, output_layer=True)
        l_real0.requires_grad(False)
        l_real1.requires_grad(False)
        self.loss_l1 = torch.clamp(self.l1(self.rec0_loss, l_real0), max=1e15) + torch.clamp(self.l1(self.rec1_loss, l_real1), max=1e15)

        # GAN loss:
        d_x = self.netdiscr(self.input)
        d_g_z = self.netdiscr(self.reconstructed)
        d_g_zp = self.netdiscr(self.generated)
        # Smooth labels
        outputs_dis = torch.cat((d_x, d_g_z, d_g_zp), dim=0)
        targets = torch.cat((torch.rand(d_x.shape[0]) * 0.19 + 0.8,
                             torch.rand(d_g_z.shape[0] + d_g_zp.shape[0]) * 0.2), dim=0)
        if torch.cuda.is_available():
            outputs_dis = outputs_dis.cuda()
            targets = targets.cuda()
        self.loss_dis = self.bce(outputs_dis, targets)
        self.loss_gen = - self.loss_dis

        # Disable training of either decoder or discriminator if optimization becomes unbalanced
        # (as measured by comparing to some predefined bounds) (as in https://github.com/lucabergamini/VAEGAN-PYTORCH/blob/master/main.py)

        if torch.mean(d_x).data.item() > equilibrium + margin and \
                torch.mean(d_g_z) < equilibrium - margin and torch.mean(d_g_zp) < equilibrium + margin:
            self.train_dis = False
        else:
            self.train_dis = True

        if torch.mean(d_x).data.item() < equilibrium - margin and \
                torch.mean(d_g_z) > equilibrium + margin and torch.mean(d_g_zp) > equilibrium + margin:
            self.train_dec = False
        else:
            self.train_dec = True
    
    def backward(self):
        gamma = 1.0
        # Optimize:
        self.optimizers['opt_enc'].zero_grad()
        loss_enc = self.loss_kld + self.loss_l1
        loss_enc.backward(keep_graph=True)  # since l1 loss is used again in decoder optimization
        self.optimizers['opt_enc'].step()  # encoder
        # del self.loss_kld  # release graph on unneeded loss

        loss_dec = gamma * self.loss_l1 + self.loss_gen
        if self.train_dec:
            self.optimizers['opt_dec'].zero_grad()
            loss_dec.backward(retain_graph=True)  # since gan loss is used again
            self.optimizers['opt_dec'].step()  # decoder

            # del loss_enc, self.loss_l1  # release graph on unneeded loss

        if self.train_dis:
            self.optimizers['opt_dis'].zero_grad()
            self.loss_dis.backward(retain_graph=True)
            self.optimizers['opt_dis'].step()  # discriminator

        # if train_dec: del loss_dec
        # del self.loss_dis, self.generated, l_rec

        # Optimize encoder w.r. to l1 loss for reconstruction (calculated later to save gpu space)

        self.optimizers['opt_vae'].zero_grad()
        rec_loss = torch.sum(l1_rec(self.reconstructed, inputs) * gts_weight)
        vae_loss = rec_loss + self.loss_l1  # I added l1 loss here ...
        vae_loss.backward()
        self.optimizers['opt_vae'].step()
        


########## NETWORKS ###########
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()
        self.iscuda = torch.cuda.is_available()

    def forward(self, input):
        mu = input[0]
        logvar = input[1]

        std = logvar.mul(0.5).exp_() #calculate the STDEV
        if self.iscuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()  #random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_()  #random normalized noise
        return eps.mul(std).add_(mu)


class _Encoder(nn.Module):
    def __init__(self, image_size, num_filt_gen, num_lat_dim, num_channels=3):
        super(_Encoder, self).__init__()

        n = math.log2(image_size)

        assert n==round(n),'image_size must be a power of 2'
        assert n>=3,'image_size must be at least 8'
        n=int(n)

        num_lat_dim = max(num_lat_dim // (8*8), 1) * 8 * 8 #ensure multiple of 16

        self.encoder = nn.Sequential()
        self.encoder.add_module('input-conv', nn.Conv2d(num_channels, num_filt_gen, 3, 1, 0))
        self.encoder.add_module('input-relu', nn.LeakyReLU(inplace=False))
        for i in range(n-4):
            #Convolutions have stride 2 !
            self.encoder.add_module('conv_{}'.format(i),
                nn.Conv2d(num_filt_gen*2**(i), num_filt_gen * 2**(i+1), 3, 1, padding=0))
            self.encoder.add_module('dropout_{}'.format(i), nn.Dropout(p=0.3))
            self.encoder.add_module('norm_{}'.format(i), nn.InstanceNorm2d(num_filt_gen * 2**(i+1)))
            self.encoder.add_module('relu_{}'.format(i), nn.LeakyReLU(inplace=False))
            self.encoder.add_module('pool{}'.format(i), nn.AvgPool2d(2, stride=2))
        #Output is 8x8
        self.encoder.add_module('output-conv0', nn.Conv2d(num_filt_gen * 2**(n-4), num_lat_dim // (8*8), 3, 1, 0))
        self.encoder.add_module('output_norm0', nn.InstanceNorm2d(num_lat_dim // (8*8)))
        self.encoder.add_module('dropout0', nn.Dropout(p=0.3))
        #total number of outputs is == num_lat_dim

        self.intermediate0 = nn.Linear(num_lat_dim, num_lat_dim)
        self.instnorm = nn.LayerNorm(num_lat_dim, elementwise_affine=False)
        self.relu = nn.LeakyReLU(inplace=False)

        #Compute std and variance
        self.means = nn.Linear(num_lat_dim, num_lat_dim * 2)
        self.varn = nn.Linear(num_lat_dim, num_lat_dim * 2)

    def forward(self, input):
        output = nn.functional.interpolate(self.encoder(input), (8,8))
        inter0 = self.relu(self.instnorm(self.intermediate0(output.view(output.size(0), -1))))
        mu = self.means(inter0)
        sig = self.varn(inter0)
        return mu.view(mu.size(0), mu.size(1) // (8*8), 8, 8), sig.view(sig.size(0), sig.size(1) // (8*8), 8, 8)


class _Decoder(nn.Module):
    def __init__(self, image_size, num_filt_gen, num_lat_dim, num_channels=3):
        super(_Decoder, self).__init__()

        n = math.log2(image_size)

        assert n==round(n),'image_size must be a power of 2'
        assert n>=3,'image_size must be at least 8'
        n=int(n)

        num_lat_dim = max(num_lat_dim // (8*8), 1) * 8 * 8 #ensure multiple of 16

        self.input_block = nn.Sequential(
            nn.Upsample(size=(16, 16), mode="bilinear"),
            nn.Conv2d(num_lat_dim * 2 // (8 * 8), num_filt_gen * 2 ** (n - 4), 3, 1, padding=1),
            nn.InstanceNorm2d(num_filt_gen * 2 ** (n - 4), affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv2d(num_filt_gen * 2 ** (n - 4), num_filt_gen * 2 ** (n - 4), 3, 1, padding=1),
            nn.InstanceNorm2d(num_filt_gen * 2 ** (n - 4), affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.3)
            )

        self.upscale = nn.Sequential(nn.Upsample(size=(16, 16), mode="bilinear"))
        for i in range(n-4, 0, -1):
            self.upscale.add_module('conv_{}'.format(i), nn.Conv2d(num_filt_gen * 2**i,
                                                                  num_filt_gen * 2**(i-1), 3, 1, padding=1))
            self.upscale.add_module('dropout{}'.format(i), nn.Dropout(p=0.3))
                        #output_padding=1 specifies correct size for 3x3 convolution kernel with stride 2
            self.upscale.add_module('norm_{}'.format(i), nn.InstanceNorm2d(num_filt_gen * 2**(i-1), affine=False))
            self.upscale.add_module('relu_{}'.format(i), nn.LeakyReLU(inplace=True))
            self.upscale.add_module('upsample_{}'.format(i), nn.Upsample(scale_factor=2, mode="bilinear"))

        self.final = nn.Sequential(nn.Conv2d(num_filt_gen, num_filt_gen, 2, 1, padding=0),
                                   nn.InstanceNorm2d(num_filt_gen, track_running_stats=True),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Dropout(p=0.3),
                                   nn.Conv2d(num_filt_gen, num_filt_gen, 2, 1, padding=0),
                                   nn.InstanceNorm2d(num_filt_gen, track_running_stats=True),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Dropout(p=0.1),
                                   nn.Conv2d(num_filt_gen, num_channels, 1, 1, padding=1),
                                   nn.Sigmoid()
                                   )

    def forward(self, inputs):
        inputs = self.input_block(inputs)
        upscaled = self.upscale(inputs)
        final = self.final(upscaled)
        return final


class VAE(nn.Module):
    def __init__(self, image_size, ngpu, num_filt_gen, num_lat_dim, num_channels=3):
        super(self.netvae, self).__init__()
        self.ngpu = ngpu
        self.sampler = _Sampler()
        self.iscuda = torch.cuda.is_available() and ngpu > 0
        self.nz = num_lat_dim #to create noise for later
        self.image_size = image_size

        self.encoder = _Encoder(image_size, num_filt_gen, num_lat_dim, num_channels=num_channels)
        self.decoder = _Decoder(image_size, num_filt_gen, num_lat_dim, num_channels=num_channels)

    def forward(self, input):
        #Reconstruct
        self.encoded = self.encoder(input)
        sampled = self.sampler(self.encoded)
        N = self.encoded[0].size(0)
        self.reconstructed = self.decoder(sampled)

        # Sample from prior
        noise = torch.normal(torch.zeros((N, self.nz * 2 // (8*8), 8, 8)), torch.ones((N, self.nz * 2 // (8*8), 8, 8))) #feeding a random latent vector to decoder # does this encourage repr for glands away from z=0 ???
        if self.iscuda: noise = noise.cuda()
        self.generated = self.decoder(noise)
        return self.encoded, self.reconstructed, self.generated

    def sample_from(self, mu, logvar):
        sampled = self.sampler((mu, logvar))
        self.reconstructed = self.decoder(sampled)
        return self.reconstructed, sampled


class Discriminator(nn.Module):
    def __init__(self, image_size, ngpu, num_filt_discr, num_channels=3):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.iscuda = torch.cuda.is_available() and ngpu > 0

        n = math.log2(image_size)
        assert n==round(n),'image_size must be a power of 2'
        assert n>=3,'image_size must be at least 8'
        n=int(n)
        m = n - 4

        #Main downsamples 3 times
        self.main = nn.Sequential(nn.Conv2d(num_channels, num_filt_discr * 2 ** (0), 3, 1, 0))
        self.main.add_module('input-relu', nn.LeakyReLU(inplace=False))

        for i in range(m):
            # Convolutions have stride 2 !
            self.main.add_module('conv_{}'.format(i),
                                    nn.Conv2d(num_filt_discr * 2 ** (i), num_filt_discr * 2 ** (i + 1), 3, 1, padding=1))
            self.main.add_module('dropout_{}'.format(i), nn.Dropout(p=0.3))
            self.main.add_module('norm_{}'.format(i), nn.InstanceNorm2d(num_filt_discr * 2 ** (i + 1), affine=False))
            self.main.add_module('relu_{}'.format(i), nn.LeakyReLU(inplace=False))
            self.main.add_module('pool{}'.format(i), nn.AvgPool2d(2, stride=2))


        # Feature map used for style loss
        self.features0 = nn.Sequential(
            nn.Conv2d(num_filt_discr * 2 ** (m + 1 -1 ), num_filt_discr * 2 ** (m + 2), 3),
            nn.InstanceNorm2d(num_filt_discr * 2 ** (m + 2), affine=False),
            nn.LeakyReLU(inplace=False),
            nn.AvgPool2d(2, stride=2)
        )

        #Feature map used for style loss
        self.features1 = nn.Sequential(
            nn.Conv2d(num_filt_discr * 2 ** (m + 2), num_filt_discr * 2 ** (m + 3), 3, padding=1),
            nn.InstanceNorm2d(num_filt_discr * 2 ** (m + 3), affine=False),
            nn.LeakyReLU(inplace=False)
        )

        #TODO could use more than one layer and sum MSEs

        self.end = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(num_filt_discr * 2 ** (m + 3), num_filt_discr * 2 ** (m + 4), 3, padding=1),
            nn.InstanceNorm2d(num_filt_discr * 2 ** (m + 4), affine=False),
            nn.LeakyReLU(inplace=False),
            nn.AvgPool2d(2, stride=2)
        )

        #Classifier
        self.classifier = nn.Sequential(
                        nn.Linear(num_filt_discr * 2 ** (m + 4) * 1 * 1, 300), #padding issues
                        nn.LeakyReLU(inplace=False),
                        nn.Linear(300, 1),
                        nn.Sigmoid()
                        )

        self.aux_classifier = nn.Sequential(
            nn.Linear(num_filt_discr * 2 ** (m + 4) * 1 * 1, 300),  # padding issues
            nn.LeakyReLU(inplace=False),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

    def forward(self, input, output_layer=False):
        output = self.main(input)
        features0 = self.features0(output)
        features1 = self.features1(features0)
        if output_layer:
            return features0, features1  #return layer activation
        output = self.end(features1)
        output = output.view(output.size(0), -1)  #reshape tensor
        y = self.classifier(output)  #return class probability
        c = self.aux_classifier(output)
        return y, c
