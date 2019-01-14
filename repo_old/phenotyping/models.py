import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

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
        eps = Variable(eps)
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
        super(VAE, self).__init__()
        self.ngpu = ngpu
        self.sampler = _Sampler()
        self.iscuda = torch.cuda.is_available() and ngpu > 0
        self.nz = num_lat_dim #to create noise for later
        self.image_size = image_size

        self.encoder = _Encoder(image_size, num_filt_gen, num_lat_dim, num_channels=num_channels)
        self.decoder = _Decoder(image_size, num_filt_gen, num_lat_dim, num_channels=num_channels)

    def forward(self, input):
        #Reconstruct
        encoded = self.encoder(input)
        sampled = self.sampler(encoded)
        N = encoded[0].size(0)
        reconstructed = self.decoder(sampled)

        # Sample from prior
        noise = torch.normal(torch.zeros((N, self.nz * 2 // (8*8), 8, 8)), torch.ones((N, self.nz * 2 // (8*8), 8, 8))) #feeding a random latent vector to decoder # does this encourage repr for glands away from z=0 ???
        if self.iscuda: noise = noise.cuda()
        generated = self.decoder(noise)
        return encoded, reconstructed, generated

    def sample_from(self, mu, logvar):
        sampled = self.sampler((mu, logvar))
        reconstructed = self.decoder(sampled)
        return reconstructed, sampled


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
