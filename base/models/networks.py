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


# Helper functions

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=tuple()):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.nepoch) / float(opt.nepoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nepoch, eta_min=0)
    elif opt.lr_policy == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=opt.learning_rate / 10, max_lr=opt.learning_rate,
                             step_size=opt.nepoch_decay, mode='triangular2')
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# Helper classes

class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.

    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs















