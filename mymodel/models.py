#Segmentation networks

import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers=[]
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout: layers.append(nn.Dropout(0.1, inplace=False)) #cannot be inplace as need gradient
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(0.2, inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.decode(x)
        x = F.upsample(x, [dim * 2 for dim in x.shape[2:]], mode='bilinear')
        return x

class _DecoderBlock2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        """
        :param in_channels:
        :param middle_channels:
        :param out_channels:
        :param dropout:

        2 convolutional layers instead of 3
        """
        super(_DecoderBlock2, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, inplace=False),
        )

    def forward(self, x):
        x = self.decode(x)
        x = F.upsample(x, [dim * 2 for dim in x.shape[2:]], mode='bilinear')
        return x


class UNet1(nn.Module):
    def __init__(self, num_classes, num_channels=64):
        super(UNet1, self).__init__()
        self.enc1 = _EncoderBlock(3, num_channels)
        self.enc2 = _EncoderBlock(num_channels, num_channels*2)
        self.enc3 = _EncoderBlock(num_channels*2, num_channels*4, dropout=True)
        self.enc4 = _EncoderBlock(num_channels*4, num_channels*4, dropout=True)
        self.enc5 = _EncoderBlock(num_channels*4, num_channels*8, dropout=True)
        self.center = _DecoderBlock(num_channels*8, num_channels*16, num_channels*8)
        self.dec5 = _DecoderBlock(num_channels*16, num_channels*8, num_channels*4, dropout=True)
        self.dec4 = _DecoderBlock(num_channels*8, num_channels*8, num_channels*4, dropout=True)
        self.dec3 = _DecoderBlock(num_channels*8, num_channels*4, num_channels*2)
        self.dec2 = _DecoderBlock(num_channels*4, num_channels*4, num_channels)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=3),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3),
            nn.InstanceNorm2d(num_channels),
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

    def __init__(self, num_classes, num_channels=32):
        super(UNet2, self).__init__()
        assert(num_channels % 2 == 0)
        self.enc1 = _EncoderBlock(3, num_channels)
        self.enc2 = _EncoderBlock(num_channels, num_channels*2)
        self.enc3 = _EncoderBlock(num_channels*2, num_channels*4)
        self.enc4 = _EncoderBlock(num_channels*4, num_channels*8, dropout=True)
        self.enc5 = _EncoderBlock(num_channels*8, num_channels*16, dropout=True)
        self.enc6 = _EncoderBlock(num_channels*16, num_channels*16, dropout=True)
        self.center = _DecoderBlock(num_channels*16, num_channels*32, num_channels*16)
        self.dec6 = _DecoderBlock(num_channels*32, num_channels*16, num_channels*16, dropout=True)
        self.dec5 = _DecoderBlock(num_channels*32, num_channels*16, num_channels*8, dropout=True)
        self.dec4 = _DecoderBlock(num_channels*16, num_channels*8, num_channels*4, dropout=True)
        self.dec3 = _DecoderBlock(num_channels*8, num_channels*4, num_channels*2)
        self.dec2 = _DecoderBlock(num_channels*4, num_channels*2, num_channels)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=3),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
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


class UNet3(nn.Module):

    def __init__(self, num_classes, num_channels=32):
        super(UNet3, self).__init__()
        self.enc1 = _EncoderBlock(3, num_channels*2)
        self.enc2 = _EncoderBlock(num_channels*2, num_channels*2)
        self.enc3 = _EncoderBlock(num_channels*2, num_channels*4)
        self.enc4 = _EncoderBlock(num_channels*4, num_channels*8, dropout=True)
        self.enc5 = _EncoderBlock(num_channels*8, num_channels*16, dropout=True)
        self.enc6 = _EncoderBlock(num_channels*16, num_channels*32, dropout=True)
        self.center = _DecoderBlock2(num_channels*32, num_channels*32, num_channels*32)
        self.dec6 = _DecoderBlock2(num_channels*64, num_channels*32, num_channels*16, dropout=True)
        self.dec5 = _DecoderBlock2(num_channels*32, num_channels*16, num_channels*8, dropout=True)
        self.dec4 = _DecoderBlock2(num_channels*16, num_channels*8, num_channels*4, dropout=True)
        self.dec3 = _DecoderBlock2(num_channels*8, num_channels*4, num_channels*2)
        self.dec2 = _DecoderBlock2(num_channels*4, num_channels*4, num_channels*2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_channels*4, num_channels*4, kernel_size=3),
            nn.InstanceNorm2d(num_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels*4, num_channels*2, kernel_size=3),
            nn.InstanceNorm2d(num_channels*4),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(num_channels*2, num_classes, kernel_size=1)
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


class UNet4(nn.Module):

    def __init__(self, num_classes, num_channels=32):
        super(UNet4, self).__init__()
        self.input_block = self.dec1 = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3),
            nn.InstanceNorm2d(num_channels*2),
            nn.ReLU(inplace=True)
        )
        self.enc1 = _EncoderBlock(num_channels*2, num_channels*2)
        self.enc2 = _EncoderBlock(num_channels*2, num_channels*2)
        self.enc3 = _EncoderBlock(num_channels*2, num_channels*4)
        self.enc4 = _EncoderBlock(num_channels*4, num_channels*8, dropout=True)
        self.enc5 = _EncoderBlock(num_channels*8, num_channels*16, dropout=True)
        self.enc6 = _EncoderBlock(num_channels*16, num_channels*32, dropout=True)
        self.center = _DecoderBlock2(num_channels*32, num_channels*32, num_channels*32)
        self.dec6 = _DecoderBlock2(num_channels*64, num_channels*32, num_channels*16, dropout=True)
        self.dec5 = _DecoderBlock2(num_channels*32, num_channels*16, num_channels*8, dropout=True)
        self.dec4 = _DecoderBlock2(num_channels*16, num_channels*8, num_channels*4, dropout=True)
        self.dec3 = _DecoderBlock2(num_channels*8, num_channels*4, num_channels*2)
        self.dec2 = _DecoderBlock2(num_channels*4, num_channels*4, num_channels*2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_channels*4, num_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels*4, num_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_channels*2),
            nn.ReLU(inplace=True)
        )
        self.final0 = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels*2, num_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_channels*2),
            nn.ReLU(inplace=True)
        )
        self.final1 = nn.Conv2d(num_channels*2, num_classes, kernel_size=1)
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
        dec6 = self.dec6(torch.cat([center, F.upsample(enc6, center.size()[2:], mode='bilinear')], 1))
        dec5 = self.dec5(torch.cat([dec6, F.upsample(enc5, dec6.size()[2:], mode='bilinear')], 1))
        dec4 = self.dec4(torch.cat([dec5, F.upsample(enc4, dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final0 = self.final0(F.upsample(dec1, x.size()[2:], mode='bilinear'))
        final1 = self.final1(F.upsample(final0, x.size()[2:], mode='bilinear'))
        return final1
