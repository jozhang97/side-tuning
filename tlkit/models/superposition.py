import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import warnings

from tlkit.utils import forward_sequential
from .basic_models import LambdaLayer

class ProjectedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True,
                 key_pick='hash', learn_key=False):
        super(ProjectedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        # Set up weight
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Set up bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Set up projection
        o_dim = self.in_channels*self.kernel_size[0]*self.kernel_size[1]
        o = torch.from_numpy( np.random.binomial(p=0.5, n=1, size=(o_dim)).astype(np.float32) * 2 - 1 )
        self.context = nn.Parameter(o, requires_grad=learn_key).view(1, self.in_channels, self.kernel_size[0], self.kernel_size[1]).cuda()

    def forward(self, x):
        return F.conv2d(x, self.weight*self.context, self.bias, stride=self.stride, padding=self.padding)

class HashConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, period,
                 stride=1, padding=0, bias=True,
                 key_pick='hash', learn_key=False, debug=False):
        super(HashConv2d, self).__init__()
        if period == 1:
            warnings.warn('Are you sure period==1? Superposition does nothing in this case')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        # Set up weight
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Set up bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Set up projection
        if debug:
            warnings.warn('setting p=1, effectively removing psp')
            p = 1
        else:
            p = 0.5
        o_dim = self.in_channels*self.kernel_size[0]*self.kernel_size[1]
        o = torch.from_numpy( np.random.binomial(p=p, n=1, size=(o_dim, period)).astype(np.float32) * 2 - 1 )
        self.context = nn.Parameter(o, requires_grad=learn_key)

    def forward(self, x, time):
        net_time = time % self.context.shape[1]
        o = self.context[:, net_time].view(1,
                                     self.in_channels,
                                     self.kernel_size[0],
                                     self.kernel_size[1])
        return F.conv2d(x, self.weight*o, self.bias, stride=self.stride, padding=self.padding)

class HashBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, period, stride=1, option='A',
                 batchnorm_kwargs={'affine': True, 'track_running_stats':False}, debug=False):
        '''
        {option:'A', fancy_bn: True} is the default that matches BasicBlock
        {option:'B', fancy_bn: False} is the default in original repo
        '''
        super(HashBasicBlock, self).__init__()
        self.conv1 = HashConv2d(in_planes, planes, 3, period, stride=stride, padding=1, bias=False, debug=debug)
        self.bn1 = nn.BatchNorm2d(planes, **batchnorm_kwargs)
        self.conv2 = HashConv2d(planes, planes, 3, period, stride=1, padding=1, bias=False, debug=debug)
        self.bn2 = nn.BatchNorm2d(planes, **batchnorm_kwargs)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.ModuleList(
                    [HashConv2d(in_planes, self.expansion*planes, 1, period, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes, **batchnorm_kwargs)]
                )

    def forward(self, x, time):
        out = F.relu(self.bn1(self.conv1(x, time)))
        out = self.bn2(self.conv2(out, time))
        out += forward_sequential(x, self.shortcut, time)
        out = F.relu(out)
        return out


class HashBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, period=None,
                 batchnorm_kwargs={'affine': True, 'track_running_stats':False},
                 **extra_kwargs):
        super(HashBottleneck, self).__init__()
        assert period is not None, 'Need a period for psp'
        self.conv1 = HashConv2d(inplanes, planes, kernel_size=1, period=period, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, **batchnorm_kwargs)
        self.conv2 = HashConv2d(planes, planes, kernel_size=3, period=period, stride=stride, bias=False, padding=1)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, **batchnorm_kwargs)
        self.conv3 = HashConv2d(planes, planes * 4, kernel_size=1, period=period, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, **batchnorm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, time):
        residual = x

        out = self.conv1(x, time)
        out = self.bn1(out)
        out = self.relu(out)

        # out = F.pad(out, pad=(1,1,1,1), mode='constant', value=0)  # other modes are reflect, replicate
        out = self.conv2(out, time)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, time)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = forward_sequential(x, self.downsample, time)

        out += residual
        out = self.relu(out)

        return out

class BinaryHashLinear(nn.Module):
    def __init__(self, in_features, out_features, period, bias=True, key_pick='hash', learn_key=True):
        super(BinaryHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(in_features, out_features))
        rand_01 = np.random.binomial(p=.5, n=1, size=(in_features, period)).astype(np.float32)
        o = torch.from_numpy(rand_01*2 - 1)

        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.context = nn.Parameter(o)
        if not learn_key:
            self.context.requires_grad = False

        if period == 1:
            warnings.warn('Are you sure period==1? Superposition does nothing in this case')

    def forward(self, x, time):
        o = self.context[:, int(time)]
        return F.linear(x * o, self.weight, self.bias)

