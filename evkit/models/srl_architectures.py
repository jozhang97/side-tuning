import torch.nn as nn
from torch.nn import Parameter, ModuleList
import torch.nn.functional as F
import torch
import multiprocessing
import numpy as np
import os
from gym import spaces
from torchvision.models import resnet18
from evkit.rl.utils import init, init_normc_
from evkit.preprocess import transforms
import torchvision as vision
from evkit.models.architectures import FrameStacked, Flatten, atari_conv

init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              nn.init.calculate_gain('relu'))

N_CHANNELS = 3
def getNChannels():
    return N_CHANNELS

########################
# SRL
########################


class BaseModelSRL(nn.Module):
    """
    Base Class for a SRL network
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelSRL, self).__init__()
        
    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.forward(observations)

    def forward(self, x):
        raise NotImplementedError

        



class BaseModelAutoEncoder(BaseModelSRL):
    """
    Base Class for a SRL network (autoencoder family)
    It implements a getState method to retrieve a state from observations
    """
    def __init__(self, n_frames, n_map_channels=0, use_target=True, output_size=512):        
        super(BaseModelAutoEncoder, self).__init__()
        self.output_size = output_size
        self.n_frames = 4
        self.n_frames = n_frames
        self.output_size = output_size
        self.n_map_channels = n_map_channels
        self.use_target = use_target
        self.use_map = n_map_channels > 0

        if self.use_map:
            self.map_tower = nn.Sequential(
                atari_conv(self.n_frames * self.n_map_channels),
                nn.Conv2d(32, 64, kernel_size=4, stride=1), #, padding=3, bias=False),
                nn.ReLU(inplace=True),
        )

        if self.use_target:
            self.target_channels = 3
        else:
            self.target_channels = 0
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        self.encoder_conv = nn.Sequential(
            # 224x224xN_CHANNELS -> 112x112x64
            nn.Conv2d(getNChannels(), 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 13x13x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 27x27x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 55x55x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 111x111x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, getNChannels(), kernel_size=4, stride=2),  # 224x224xN_CHANNELS
        )
        self.encoder = FrameStacked(self.encoder_conv, self.n_frames)
        self.conv1 = nn.Conv2d(self.n_frames * (64 + self.target_channels), 64, 3, stride=1) # c4 s 4
        self.flatten = Flatten()
        self.fc1 = init_(nn.Linear(64 * 4 * 4 * (self.use_map) + 64 * 4 * 4 * (1), 1024))
        self.fc2 = init_(nn.Linear(1024, self.output_size))

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encode(observations)

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
#         raise NotImplementedError
        self.encoder_conv(x)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
#         raise NotImplementedError
        self.decoder_conv(x)

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        x_taskonomy = x['taskonomy']
        if self.use_target:
            x_taskonomy = torch.cat([x_taskonomy, x["target"]], dim=1)
        x_taskonomy = F.relu(self.conv1(x_taskonomy))
        if self.use_map:
            x_map = x['map']
            x_map = self.map_tower(x_map)
            x_taskonomy = torch.cat([x_map, x_taskonomy], dim=1)
        x = self.flatten(x_taskonomy)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
        encoded = self.encode(x) 
#         decoded = self.decode(encoded).view(input_shape)
        return encoded #, decoded
    
    
    
def conv3x3(in_planes, out_planes, stride=1):
    """"
    From PyTorch Resnet implementation
    3x3 convolution with padding
    :param in_planes: (int)
    :param out_planes: (int)
    :param stride: (int)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



def srl_features_transform(task_path, dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)
            
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    _rescale_thunk = transforms.rescale_centercrop_resize((3, 224, 224))
    if task_path != 'pixels_as_state':
#         net = TaskonomyEncoder().cuda()
        net = nn.Sequential(
            # 224x224xN_CHANNELS -> 112x112x64
            nn.Conv2d(getNChannels(), 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        ).cuda()

        net.eval()
        if task_path != 'None':
            checkpoint = torch.load(task_path)
#             checkpoint = {k.replace('model.encoder_conv.', ''): v for k, v in checkpoint.items() if 'encoder_conv' in k}
            checkpoint = {k.replace('model.conv_layers.', '').replace('model.encoder_conv.', ''): v for k, v in checkpoint.items() if 'encoder_conv' in k or 'conv_layers' in k}
            net.load_state_dict(checkpoint)

    def encode(x):
        if task_path == 'pixels_as_state':
            return x
        with torch.no_grad():
            return net(x)
    
    def _features_transform_thunk(obs_space):
        rescale, _ = _rescale_thunk(obs_space)
        def pipeline(x):
#             x = rescale(x).view(1, 3, 224, 224)
            x = torch.Tensor(x).cuda()
            x = encode(x)
            return x.cpu()
        if task_path == 'pixels_as_state':
            raise NotImplementedError
            return pixels_as_state_pipeline, spaces.Box(-1, 1, (8, 16, 16), dtype)
        else:
            return pipeline, spaces.Box(-1, 1, (64, 6, 6), dtype)
    
    return _features_transform_thunk

