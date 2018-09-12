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
    def __init__(self, image_size, num_filt_gen, num_lat_dim, num_channels=3, batchnorm=True):
        super(_Encoder, self).__init__()

        n = math.log2(image_size)

        assert n==round(n),'image_size must be a power of 2'
        assert n>=3,'image_size must be at least 8'
        n=int(n)

        self.encoder = nn.Sequential()
        self.encoder.add_module('input-conv',nn.Conv2d(num_channels, num_filt_gen, 3, 1, 0, bias=False))
        self.encoder.add_module('input-relu',nn.ReLU(inplace=True))
        for i in range(n-3):
            #Convolutions have stride 2 !
            self.encoder.add_module('pyramid{0}-{1}conv'.format(num_filt_gen*2**i, num_filt_gen * 2**(i+1)),
                nn.Conv2d(num_filt_gen*2**(i), num_filt_gen * 2**(i+1), 3, 1, padding=0, bias=False))
            if batchnorm:
                self.encoder.add_module('pyramid{0}batchnorm'.format(num_filt_gen * 2**(i+1)), nn.BatchNorm2d(num_filt_gen * 2**(i+1)))
            self.encoder.add_module('pyramid{0}relu'.format(num_filt_gen * 2**(i+1)), nn.ReLU(inplace=True))
            self.encoder.add_module('pool{}'.format(i), nn.MaxPool2d(2, stride=2))
        #Output is 4x4
        self.encoder.add_module('output-conv', nn.Conv2d(num_filt_gen * 2**(n-3), num_lat_dim // (4*4), 3, 1, 0, bias=False))
        #total number of outputs is == num_lat_dim

        #Compute std and variance
        self.means = nn.Linear(num_lat_dim, num_lat_dim)
        self.varn = nn.Linear(num_lat_dim, num_lat_dim)

    def forward(self, input):
        output = self.encoder(input)
        mu = self.means(output.view(output.size(0), -1))
        sig = self.varn(output.view(output.size(0), -1))
        return mu.view(mu.size(0), mu.size(1) // (4*4), 4, 4), sig.view(sig.size(0), sig.size(1) // (4*4), 4, 4)


class _netG(nn.Module):
    def __init__(self, image_size, ngpu, num_filt_gen, num_lat_dim, num_channels=3, batchnorm=True):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = _Encoder(image_size, num_filt_gen, num_lat_dim, batchnorm=batchnorm)
        self.sampler = _Sampler()
        self.iscuda = torch.cuda.is_available() and ngpu > 0

        n = math.log2(image_size)

        assert n==round(n),'image_size must be a power of 2'
        assert n>=3,'image_size must be at least 8'
        n=int(n)

        self.decoder = nn.Sequential()
        # input is Z
        self.decoder.add_module('input-conv',
                nn.ConvTranspose2d(num_lat_dim // (4*4), num_filt_gen * 2**(n-3), 3, 1, padding=0, bias=False))
        if batchnorm:
            self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(num_filt_gen * 2**(n-3)))
        self.decoder.add_module('input-relu', nn.ReLU(inplace=True))

        for i in range(n-3, 0, -1):
            self.decoder.add_module('pyramid{0}-{1}conv'.format(num_filt_gen*2**i, num_filt_gen * 2**(i-1)),
                        nn.ConvTranspose2d(num_filt_gen * 2**i, num_filt_gen * 2**(i-1), 3, 2, padding=0, output_padding=1, bias=False))
                        #output_padding=1 specifies correct size for 3x3 convolution kernel with stride 2
            if batchnorm:
                self.decoder.add_module('pyramid{0}batchnorm'.format(num_filt_gen * 2**(i-1)), nn.BatchNorm2d(num_filt_gen * 2**(i-1)))
            self.decoder.add_module('pyramid{0}relu'.format(num_filt_gen * 2**(i-1)), nn.ReLU(inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(num_filt_gen, num_channels, 3, 1, padding=0, bias=False))
        self.decoder.add_module('output-sigmoid', nn.Sigmoid()) #multiplied by 255 (0 to 255 is image range) in forward

        #where is the upsampling done in the decoder ????

    def forward(self, input):
        if self.iscuda and self.ngpu > 1:
            mu, sig = nn.parallel.data_parallel(self.encoder, input, list(range(self.ngpu)))
            z = nn.parallel.data_parallel(self.sampler, [mu, sig], list(range(self.ngpu)))
            output = nn.parallel.data_parallel(self.decoder, z, list(range(self.ngpu)))
        else:
            mu, sig = self.encoder(input)
            z = self.sampler([mu, sig])
            output = self.decoder(z)
        output *= 255 #OUTPUT IS FROM 0 TO 1 OTHERWISE !!! ORIGINAL OUTPUT WAS PROBABLY NORMALIZED FROM 0 TO 1 !!!
        return output


class _netD(nn.Module):
    def __init__(self, image_size, ngpu, num_filt_discr, num_channels=3, batchnorm=True):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.iscuda = torch.cuda.is_available() and ngpu > 0
        n = math.log2(image_size)

        assert n==round(n),'image_size must be a power of 2'
        assert n>=3,'image_size must be at least 8'
        n=int(n)
        self.main = nn.Sequential()

        self.main.add_module('input-conv', nn.Conv2d(num_channels, num_filt_discr, 3, bias=False))
        self.main.add_module('relu', nn.ReLU(inplace=True))

        for i in range(n-3):
            self.main.add_module('pyramid{0}-{1}conv'.format(num_filt_discr*2**(i), num_filt_discr * 2**(i+1)), nn.Conv2d(num_filt_discr * 2 ** (i), num_filt_discr * 2 ** (i+1), 3, bias=False))
            if batchnorm:
                self.main.add_module('pyramid{0}batchnorm'.format(num_filt_discr * 2**(i+1)), nn.BatchNorm2d(num_filt_discr * 2 ** (i+1)))
            self.main.add_module('pyramid{0}relu'.format(num_filt_discr * 2**(i+1)), nn.ReLU(inplace=True))
            self.main.add_module('pool{}'.format(i), nn.MaxPool2d(2, stride=2))

        self.main.add_module('output_conv',  nn.Conv2d(num_filt_discr * 2 ** (i+1), 64, 3)) #output conv

        #Classification layer:
        self.classifier = nn.Sequential(
                        nn.Linear(4*4*64, 300), #padding issues
                        nn.ReLU(inplace=True),
                        nn.Linear(300,1),
                        nn.Sigmoid()
                        )


    def forward(self, input):
        if self.iscuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, list(range(self.ngpu)))
            output = output.view(output.size(0), -1) #reshape tensor for fc classifier
            output = nn.parallel.data_parallel(self.classifier, output, list(range(self.ngpu)))
        else:
            output = self.main(input)
            output = output.view(output.size(0), -1) #reshape tensor
            output = self.classifier(output)
        return output
