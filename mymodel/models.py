#Segmentation networks

import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, batchnorm=False):
        super(_EncoderBlock, self).__init__()
        layers=[]
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3))
        if batchnorm: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout: layers.append(nn.Dropout(0.1, inplace=False)) #cannot be inplace as need gradient
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3))
        if batchnorm: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(0.3, inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False, batchnorm=False):
        super(_DecoderBlock, self).__init__()
        layers=[]
        layers.append(nn.Conv2d(in_channels, middle_channels, kernel_size=3))
        if batchnorm: layers.append(nn.BatchNorm2d(middle_channels))
        if dropout: layers.append(nn.Dropout(0.1, inplace=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(middle_channels, middle_channels, kernel_size=3))
        if batchnorm: layers.append(nn.BatchNorm2d(middle_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout: layers.append(nn.Dropout(0.1, inplace=False))
        layers.append(nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class UNet1(nn.Module):
    def __init__(self, num_classes, num_channels=64, grayscale=False, batchnorm=False):
        super(UNet1, self).__init__()
        self.enc1 = _EncoderBlock(1 if grayscale else 3, num_channels)
        self.enc2 = _EncoderBlock(num_channels, num_channels*2)
        self.enc3 = _EncoderBlock(num_channels*2, num_channels*4, dropout=True, batchnorm=batchnorm)
        self.enc4 = _EncoderBlock(num_channels*4, num_channels*4, dropout=True, batchnorm=batchnorm)
        self.enc5 = _EncoderBlock(num_channels*4, num_channels*8, dropout=True, batchnorm=batchnorm)
        self.center = _DecoderBlock(num_channels*8, num_channels*16, num_channels*8)
        self.dec5 = _DecoderBlock(num_channels*16, num_channels*8, num_channels*4, dropout=True, batchnorm=batchnorm)
        self.dec4 = _DecoderBlock(num_channels*8, num_channels*8, num_channels*4, dropout=True, batchnorm=batchnorm)
        self.dec3 = _DecoderBlock(num_channels*8, num_channels*4, num_channels*2)
        self.dec2 = _DecoderBlock(num_channels*4, num_channels*4, num_channels)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=3),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(num_channels, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        center = self.center(enc5)
        dec5 = self.dec5(torch.cat([center, F.upsample(enc5, center.size()[2:], mode='bilinear')], 1))
        dec4 = self.dec4(torch.cat([dec5, F.upsample(enc4, dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')

class UNet2(nn.Module):
    """
    More encoders, no batchnorm, more dropout
    """
    def __init__(self, num_classes, num_channels=64, grayscale=False, batchnorm=False):
        super(UNet2, self).__init__()
        self.enc1 = _EncoderBlock(1 if grayscale else 3, num_channels)
        self.enc2 = _EncoderBlock(num_channels, num_channels*2)
        self.enc3 = _EncoderBlock(num_channels*2, num_channels*4)
        self.enc4 = _EncoderBlock(num_channels*4, num_channels*4, dropout=True, batchnorm=batchnorm)
        self.enc5 = _EncoderBlock(num_channels*4, num_channels*4, dropout=True, batchnorm=batchnorm)
        self.enc6 = _EncoderBlock(num_channels*4, num_channels*8, dropout=True, batchnorm=batchnorm)
        self.center = _DecoderBlock(num_channels*8, num_channels*16, num_channels*8)
        self.dec6 = _DecoderBlock(num_channels*16, num_channels*8, num_channels*4, dropout=True, batchnorm=batchnorm)
        self.dec5 = _DecoderBlock(num_channels*8, num_channels*4, num_channels*4, dropout=True, batchnorm=batchnorm)
        self.dec4 = _DecoderBlock(num_channels*8, num_channels*4, num_channels*4, dropout=True, batchnorm=batchnorm)
        self.dec3 = _DecoderBlock(num_channels*8, num_channels*4, num_channels*2)
        self.dec2 = _DecoderBlock(num_channels*4, num_channels*2, num_channels)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=3),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        ) #TODO assess if these batch norms affect test acc with current batch size
        self.final = nn.Conv2d(num_channels, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        center = self.center(enc6)
        dec6 = self.dec6(torch.cat([center, F.upsample(enc6, center.size()[2:], mode='bilinear')], 1))
        dec5 = self.dec5(torch.cat([dec6, F.upsample(enc5, dec6.size()[2:], mode='bilinear')], 1))
        dec4 = self.dec4(torch.cat([dec5, F.upsample(enc4, dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')
