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
            eps = torch.cuda.FloatTensor(std.size()).normal_() #random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_() #random normalized noise
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

class _Encoder(nn.Module):
    def __init__(self, image_size, num_filt_gen, num_lat_dim, num_channels=3):
        super(_Encoder, self).__init__()

        n = math.log2(image_size)

        assert n==round(n),'image_size must be a power of 2'
        assert n>=3,'image_size must be at least 8'
        n=int(n)

        num_lat_dim = max(num_lat_dim // (4*4), 1) * 4 * 4 #ensure multiple of 16

        self.encoder = nn.Sequential()
        self.encoder.add_module('input-conv', nn.Conv2d(num_channels, num_filt_gen, 3, 1, 0))
        self.encoder.add_module('input-relu', nn.ReLU(inplace=False))
        for i in range(n-3):
            #Convolutions have stride 2 !
            self.encoder.add_module('conv_{}'.format(i),
                nn.Conv2d(num_filt_gen*2**(i), num_filt_gen * 2**(i+1), 3, 1, padding=0))
            self.encoder.add_module('norm_{}'.format(i), nn.InstanceNorm2d(num_filt_gen * 2**(i+1)))
            self.encoder.add_module('relu_{}'.format(i), nn.ReLU(inplace=False))
            self.encoder.add_module('pool{}'.format(i), nn.MaxPool2d(2, stride=2))
        #Output is 4x4
        self.encoder.add_module('output-conv', nn.Conv2d(num_filt_gen * 2**(n-3), num_lat_dim // (4*4), 3, 1, 0))
        #total number of outputs is == num_lat_dim

        #Compute std and variance
        self.means = nn.Linear(num_lat_dim, num_lat_dim)
        self.varn = nn.Linear(num_lat_dim, num_lat_dim)

    def forward(self, input):
        output = self.encoder(input)
        mu = self.means(output.view(output.size(0), -1))
        sig = self.varn(output.view(output.size(0), -1))
        return mu.view(mu.size(0), mu.size(1) // (4*4), 4, 4), sig.view(sig.size(0), sig.size(1) // (4*4), 4, 4)

class VAE(nn.Module):
    def __init__(self, image_size, ngpu, num_filt_gen, num_lat_dim, num_channels=3):
        super(VAE, self).__init__()
        self.ngpu = ngpu
        self.sampler = _Sampler()
        self.encoder = _Encoder(image_size, num_filt_gen, num_lat_dim, num_channels=num_channels)
        self.iscuda = torch.cuda.is_available() and ngpu > 0
        num_lat_dim = max(num_lat_dim // (4 * 4), 1) * 4 * 4  # ensure multiple of 16
        self.nz = num_lat_dim #to create noise for later

        n = math.log2(image_size)

        assert n==round(n),'image_size must be a power of 2'
        assert n>=3,'image_size must be at least 8'
        n=int(n)

        self.decoder = nn.Sequential()
        # input is Z
        self.decoder.add_module('input-conv',
                nn.ConvTranspose2d(num_lat_dim // (4*4), num_filt_gen * 2**(n-3), 3, 1, padding=0))
        self.decoder.add_module('input-norm', nn.InstanceNorm2d(num_filt_gen * 2**(n-3)))
        self.decoder.add_module('input-relu', nn.ReLU(inplace=False))

        for i in range(n-3, 0, -1):
            self.decoder.add_module('conv_{}'.format(i),
                        nn.ConvTranspose2d(num_filt_gen * 2**i, num_filt_gen * 2**(i-1), 3, 2, padding=0, output_padding=1))
                        #output_padding=1 specifies correct size for 3x3 convolution kernel with stride 2
            self.decoder.add_module('norm_{}'.format(i), nn.InstanceNorm2d(num_filt_gen * 2**(i-1)))
            self.decoder.add_module('relu_{}'.format(i), nn.ReLU(inplace=False))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(num_filt_gen, num_channels, 3, 1, padding=0))
        self.decoder.add_module('output-sigmoid', nn.Sigmoid())

        #where is the upsampling done in the decoder ????

    def forward(self, input):
        #Reconstruct
        encoded = self.encoder(input)
        sampled = self.sampler(encoded)
        N = encoded[0].size(0)
        reconstructed = self.decoder(sampled)

        # Sample from prior
        noise = torch.normal(torch.zeros((N, self.nz // (4*4), 4, 4)), torch.ones((N, self.nz // (4*4), 4, 4))) #feeding a random latent vector to decoder # does this encourage repr for glands away from z=0 ???
        if self.iscuda: noise = noise.cuda()
        generated = self.decoder(noise)
        return encoded, reconstructed, generated


class Discriminator(nn.Module):
    def __init__(self, image_size, ngpu, num_filt_discr, num_channels=3):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.iscuda = torch.cuda.is_available() and ngpu > 0

        n = math.log2(image_size)
        assert n==round(n),'image_size must be a power of 2'
        assert n>=3,'image_size must be at least 8'
        n=int(n)

        #Main downsamples 3 times
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, num_filt_discr, 3),
            nn.InstanceNorm2d(num_filt_discr),
            nn.ReLU(inplace=False),
            nn.Conv2d(num_filt_discr, num_filt_discr*2, 3),
            nn.InstanceNorm2d(num_filt_discr*2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(num_filt_discr * 2 ** (1), num_filt_discr * 2 ** (1 + 1), 3),
            nn.InstanceNorm2d(num_filt_discr * 2 ** (1 + 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(num_filt_discr * 2 ** (2), num_filt_discr * 2 ** (2 + 1), 3),
            nn.InstanceNorm2d(num_filt_discr * 2 ** (2 + 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(num_filt_discr * 2 ** (3), num_filt_discr * 2 ** (3 + 1), 3),
            nn.InstanceNorm2d(num_filt_discr * 2 ** (3 + 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2)
        )

        #Feature map used for style loss
        self.features = nn.Sequential(
            nn.Conv2d(num_filt_discr * 2 ** (4), num_filt_discr * 2 ** (4 + 1), 3),
            nn.InstanceNorm2d(num_filt_discr * 2 ** (4 + 1)),
            nn.ReLU(inplace=False)
        )

        self.end = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(num_filt_discr * 2 ** (4+1), num_filt_discr * 2 ** (4+1), 3),
            nn.InstanceNorm2d(num_filt_discr * 2 ** (4 + 1)),
            nn.ReLU(inplace=False)
        )

        #Classifier
        self.classifier = nn.Sequential(
                        nn.Linear(4*4*num_filt_discr * 2 ** (4 + 1), 300), #padding issues
                        nn.ReLU(inplace=False),
                        nn.Linear(300,1),
                        nn.Sigmoid()
                        )

    def forward(self, input, output_layer=False):
        output = self.main(input)
        features = self.features(output)
        if output_layer:
            return features  #return layer activation
        output = self.end(features)
        output = output.view(output.size(0), -1) #reshape tensor
        y = self.classifier(output)  #return class probability
        return y
