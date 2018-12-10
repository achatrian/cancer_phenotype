#Segmentation networks
import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler, Optimizer
from torch.nn import init


def on_cluster():
    import socket, re
    hostname = socket.gethostname()
    match1 = re.search("jalapeno(\w\w)?.fmrib.ox.ac.uk", hostname)
    match2 = re.search("cuda(\w\w)?.fmrib.ox.ac.uk", hostname)
    match3 = re.search("login(\w\w)?.cluster", hostname)
    match4 = re.search("gpu(\w\w)?", hostname)
    match5 = re.search("compG(\w\w\w)?", hostname)
    match6 = re.search("rescomp(\w)?", hostname)
    return bool(match1 or match2 or match3 or match4 or match5)

if on_cluster():
    sys.path.append(os.path.expanduser('~') + '/cancer_phenotype')
else:
    sys.path.append(os.path.expanduser('~') + '/Documents/Repositories/cancer_phenotype')


# networks


class _EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(0.3, inplace=False),  # cannot be inplace as need gradient
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(0.3, inplace=False),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.in_out = (in_channels, out_channels)  # for debugging

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upsample=True):
        """
        :param in_channels:
        :param middle_channels:
        :param out_channels:
        :param dropout:

        2 convolutional layers instead of 3
        """
        super(_DecoderBlock, self).__init__()
        self.decode = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(middle_channels),
            torch.nn.Dropout(0.3, inplace=False),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(0.3, inplace=False),
        )
        self.upsample = upsample
        self.in_out = (in_channels, out_channels)  # for debugging

    def forward(self, x):
        x = self.decode(x)
        if self.upsample:
            x = F.interpolate(x, [dim * 2 for dim in x.shape[2:]], mode='bilinear')
        return x


class UNet4(torch.nn.Module):

    def __init__(self, num_classes, num_filters=32):
        super(UNet4, self).__init__()
        self.input_block = self.dec1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, num_filters, kernel_size=3),
            torch.nn.InstanceNorm2d(num_filters),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(num_filters, num_filters*2, kernel_size=3),
            torch.nn.InstanceNorm2d(num_filters*2),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.enc1 = _EncoderBlock(num_filters*2, num_filters*2)
        self.enc2 = _EncoderBlock(num_filters*2, num_filters*2)
        self.enc3 = _EncoderBlock(num_filters*2, num_filters*4)
        self.enc4 = _EncoderBlock(num_filters*4, num_filters*8, dropout=True)
        self.enc5 = _EncoderBlock(num_filters*8, num_filters*16, dropout=True)
        self.enc6 = _EncoderBlock(num_filters*16, num_filters*32, dropout=True)
        self.center = _DecoderBlock(num_filters*32, num_filters*32, num_filters*32)
        self.dec6 = _DecoderBlock(num_filters*64, num_filters*32, num_filters*16, dropout=True)
        self.dec5 = _DecoderBlock(num_filters*32, num_filters*16, num_filters*8, dropout=True)
        self.dec4 = _DecoderBlock(num_filters*16, num_filters*8, num_filters*4, dropout=True)
        self.dec3 = _DecoderBlock(num_filters*8, num_filters*4, num_filters*2)
        self.dec2 = _DecoderBlock(num_filters*4, num_filters*4, num_filters*2)
        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv2d(num_filters*4, num_filters*4, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(num_filters*4),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(num_filters*4, num_filters*2, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(num_filters*2),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final0 = torch.nn.Sequential(
            torch.nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(num_filters*2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(num_filters*2),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final1 = torch.nn.Conv2d(num_filters*2, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        input = self.input_block(x)
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        center = self.center(enc6)
        dec6 = self.dec6(torch.cat([center, F.interpolate(enc6, center.size()[2:], mode='bilinear')], 1))
        dec5 = self.dec5(torch.cat([dec6, F.interpolate(enc5, dec6.size()[2:], mode='bilinear')], 1))
        dec4 = self.dec4(torch.cat([dec5, F.interpolate(enc4, dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final0 = self.final0(F.interpolate(dec1, x.size()[2:], mode='bilinear'))
        final1 = self.final1(F.interpolate(final0, x.size()[2:], mode='bilinear'))
        return final1


class UNet(torch.nn.Module):

    def __init__(self, depth, num_classes, num_input_channels=3, num_filters=10, tile_size=512, max_multiple=32, multiples=None):
        super(UNet, self).__init__()

        self.depth = depth  # number of downsamplings / max depth of encoder network
        self.tile_size = tile_size

        self.input_block = self.dec1 = torch.nn.Sequential(
            torch.nn.Conv2d(num_input_channels, num_filters, kernel_size=3),
            torch.nn.InstanceNorm2d(num_filters),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(num_filters, num_filters, kernel_size=3),
            torch.nn.InstanceNorm2d(num_filters),
            torch.nn.LeakyReLU(inplace=True)
        )

        if not multiples:
            # standard multiples
            ndouble = min(round(math.log2(max_multiple)), depth)
            multiples = [1] * (depth - ndouble) + [2 ** d for d in range(0, ndouble + 1)]  # |multiples| = depth + 1
            # encoder multiples - first ascending decoder has max_multiple * 2 input dimension
        else:
            if len(multiples) != depth + 1:
                raise ValueError("Given multiples are less than desired # of layers ({} != {})".format(
                    len(multiples), depth + 1
                ))
        self.multiples = multiples

        for d in range(depth):
            # Build encoders
            enc = _EncoderBlock(multiples[d] * num_filters, multiples[d + 1] * num_filters)
            setattr(self, 'enc{}'.format(d), enc)
            dec = _DecoderBlock(2 * multiples[d + 1] * num_filters, 2 * multiples[d + 1] * num_filters,
                                multiples[d] * num_filters)
            setattr(self, 'dec{}'.format(d), dec)

        self.dec0.upsample = False  # center upsamples input - so output decoder must not (or output will be too large)
        self.center = _DecoderBlock(multiples[depth] * num_filters, multiples[depth] * num_filters,
                                    multiples[depth] * num_filters)

        self.output_block = torch.nn.Sequential(
            torch.nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(num_filters),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(num_filters),
            torch.nn.LeakyReLU(inplace=True)
        )

        self.final_conv = torch.nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.input_block(x)
        encoded = []
        for d in range(self.depth):
            enc = getattr(self, 'enc{}'.format(d))
            encoded.append(enc(x))
            x = encoded[-1]
        x = self.center(x)

        for d in range(self.depth-1, -1, -1):
            dec = getattr(self, 'dec{}'.format(d))
            x = torch.cat([x, F.interpolate(encoded[d], x.size()[2:], mode='bilinear')], 1)
            x = dec(x)

        x = F.interpolate(self.output_block(x), (self.tile_size,) * 2, mode='bilinear')
        y = self.final_conv(x)
        return y








