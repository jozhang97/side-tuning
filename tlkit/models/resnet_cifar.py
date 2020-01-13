'''
ADAPTED FROM: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

import numpy as np
import warnings
from tlkit.utils import forward_sequential
from .superposition import HashBasicBlock
from tlkit.models.basic_models import EvalOnlyModel
from .basic_models import LambdaLayer


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', batchnorm_kwargs={'track_running_stats': True}):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, **batchnorm_kwargs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, **batchnorm_kwargs)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes, **batchnorm_kwargs)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(EvalOnlyModel):
    def __init__(self, block, num_blocks, num_classes=10, period=None, debug=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.in_planes = 16
        self.num_classes = num_classes
    
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, period=period, debug=debug)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, period=period, debug=debug)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, period=period, debug=debug)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, period, debug):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            try:
                layers.append(block(self.in_planes, planes, period=period, stride=stride, debug=debug))
            except TypeError:
                layers.append(block(self.in_planes, planes, stride=stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)

def resnet1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

#### new stuff
class ResnetiCifar44(ResNet):
    def __init__(self, bsp=False, **new_kwargs):
        if bsp:
            super().__init__(HashBasicBlock, [7, 7, 7], **new_kwargs)
        else:
            super().__init__(BasicBlock, [7, 7, 7], **new_kwargs)
        self.bsp = bsp

    def forward(self, x, task_idx:int=-1):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.bsp:  # bsp mode
            out = forward_sequential(out, self.layer1, task_idx)
            out = forward_sequential(out, self.layer2, task_idx)
            out = forward_sequential(out, self.layer3, task_idx)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResnetiCifar(ResNet):
    def __init__(self, **new_kwargs):
        super().__init__(BasicBlock, [3, 3, 3], **new_kwargs)

    def forward(self, x):
        return super().forward(x)


class ResnetiCifar44NoLinear(ResNet):
    def __init__(self, bsp=False, final_act=True, **new_kwargs):
        if bsp:
            super().__init__(HashBasicBlock, [7, 7, 7], **new_kwargs)
        else:
            super().__init__(BasicBlock, [7, 7, 7], **new_kwargs)
        self.bsp = bsp
        self.final_act = final_act
        del self.linear

    def forward(self, x, time:int=-1):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.bsp:  # bsp mode
            out = forward_sequential(out, self.layer1, time)
            out = forward_sequential(out, self.layer2, time)
            out = forward_sequential(out, self.layer3, time)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            if self.final_act:
                out = self.layer3(out)
            else:
                for i in range(6):
                    out = self.layer3[i](out)

                basic_block = self.layer3[6]
                block_input = out
                out = F.relu(basic_block.bn1(basic_block.conv1(out)))
                # out = basic_block.bn2(basic_block.conv2(out))
                out = basic_block.conv2(out)
                out += basic_block.shortcut(block_input)
                # out = F.relu(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

    def start_training(self):
        if self.bsp:
            warnings.warn('Before training: Applying context to weights, Are you sure?')
            for name, param in self.named_parameters():
                if 'conv' in name and 'weight' in name and 'layer' in name:
                    o = torch.from_numpy( np.random.binomial(p=0.5, n=1, size=param.shape[1:]).astype(np.float32) * 2 - 1 ).cuda()
                    self.state_dict()[name] = param * o


class ResnetiCifar44NoLinearWithCache(ResnetiCifar44NoLinear):
    def forward(self, x, time:int=-1):
        self.x_pre_layer1 = F.relu(self.bn1(self.conv1(x)))  # (16,32,32)
        self.x_layer1 = self.layer1(self.x_pre_layer1)       # (16,32,32)
        self.x_layer2 = self.layer2(self.x_layer1)           # (32,16,16)
        self.x_layer3 = self.layer3(self.x_layer2)           # (64, 8, 8)
        out = F.avg_pool2d(self.x_layer3, self.x_layer3.size()[3])
        out = out.view(out.size(0), -1)
        return out, [self.x_pre_layer1.detach(), self.x_layer1.detach(), self.x_layer2.detach(), self.x_layer3.detach()]

