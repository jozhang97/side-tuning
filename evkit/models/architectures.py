import torch.nn as nn
from torch.nn import Parameter, ModuleList
import torch.nn.functional as F
import torch
import multiprocessing
import numpy as np
import os
from gym import spaces
from torchvision.models import resnet18
import torchvision as vision
import warnings

from evkit.rl.utils import init, init_normc_
from evkit.preprocess import transforms

init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              nn.init.calculate_gain('relu'))

def atari_nature(num_inputs, num_outputs=512):
    init_ = lambda m: init(m,
                  nn.init.orthogonal_,
                  lambda x: nn.init.constant_(x, 0),
                  nn.init.calculate_gain('relu'))

    return nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, num_outputs)), # 512 original outputs
            nn.ReLU()
    )

def atari_conv(num_inputs):
    init_ = lambda m: init(m,
                  nn.init.orthogonal_,
                  lambda x: nn.init.constant_(x, 0),
                  nn.init.calculate_gain('relu'))

    return nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU())

def atari_small_conv(num_inputs):
    init_ = lambda m: init(m,
                  nn.init.orthogonal_,
                  lambda x: nn.init.constant_(x, 0),
                  nn.init.calculate_gain('relu'))

    return nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU())

def atari_match_conv(num_frames, num_inputs_per_frame):
    # Expected Input size: (3*N, 84, 84)
    # Expected Output size: (8*N, 16, 16)
    num_inputs = num_frames * num_inputs_per_frame
    init_ = lambda m: init(m,
                           nn.init.orthogonal_,
                           lambda x: nn.init.constant_(x, 0),
                           nn.init.calculate_gain('relu'))
    return nn.Sequential(
        init_(nn.Conv2d(num_inputs, 64, 8, stride=4)),
        nn.ReLU(),
        init_(nn.Conv2d(64, 8 * num_frames, 5, stride=1)),
        nn.ReLU())

def atari_big_conv(num_frames, num_inputs_per_frame):
    # Expected Input size: (3*N, 256, 256)
    # Expected Output size: (8*N, 16, 16)
    # TODO get the dimenions here working
    num_inputs = num_frames * num_inputs_per_frame
    init_ = lambda m: init(m,
                           nn.init.orthogonal_,
                           lambda x: nn.init.constant_(x, 0),
                           nn.init.calculate_gain('relu'))
    return nn.Sequential(
        init_(nn.Conv2d(num_inputs, 64, 8, stride=4)),
        nn.ReLU(),
        init_(nn.Conv2d(64,  64, 5, stride=1)),
        nn.ReLU(),
        init_(nn.Conv2d(64, 8 * num_frames, 5, stride=1)),
        nn.ReLU()
    )

def atari_nature_vae(num_inputs, num_outputs=512):
    init_ = lambda m: init(m,
                  nn.init.orthogonal_,
                  lambda x: nn.init.constant_(x, 0),
                  nn.init.calculate_gain('relu'))
    
    nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, num_outputs)),
            nn.ReLU()
    )

    
def is_cuda(model):
    return next(model.parameters()).is_cuda



def task_encoder(checkpoint_path):
    net = TaskonomyEncoder()
    net.eval()
    print(checkpoint_path)
    if checkpoint_path != None:
        path_pth_ckpt = os.path.join(checkpoint_path)
        checkpoint = torch.load(path_pth_ckpt)
        net.load_state_dict(checkpoint['state_dict'])
    return net


class AtariNet(nn.Module):
    def __init__(self, n_frames,
                 n_map_channels=0,
                 use_target=True,
                 output_size=512):
        super(AtariNet, self).__init__()
        self.n_frames = n_frames
        self.use_target = use_target
        self.use_map = n_map_channels > 0
        self.map_channels = n_map_channels
        self.output_size = output_size
        
        if self.use_map:
            self.map_tower = atari_conv(num_inputs=self.n_frames * self.map_channels)
        else:
            self.map_channels = 0
        if self.use_target:
            self.target_channels = 3
        else:
            self.target_channels = 0

        self.image_tower = atari_small_conv(num_inputs=self.n_frames*3)
        self.conv1 = nn.Conv2d(64 + (self.n_frames * self.target_channels), 32, 3, stride=1)
        self.flatten = Flatten()
        self.fc1 = init_(nn.Linear(32 * 7 * 7 * (self.use_map + 1), 1024))
        self.fc2 = init_(nn.Linear(1024, self.output_size))

    def forward(self, x):
        x_rgb = x['rgb_filled']
        x_rgb = self.image_tower(x_rgb)
        if self.use_target:
            x_rgb = torch.cat([x_rgb, x["target"]], dim=1)
        x_rgb = F.relu(self.conv1(x_rgb))
        if self.use_map:
            x_map = x['map']
            x_map = self.map_tower(x_map)
            x_rgb = torch.cat([x_rgb, x_map], dim=1)
        x = self.flatten(x_rgb)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class AtariNatureEncoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super().__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels
        
        init_ = lambda m: init(m,
                  nn.init.orthogonal_,
                  lambda x: nn.init.constant_(x, 0),
                  nn.init.calculate_gain('relu'))
        
        self.conv1 = init_(nn.Conv2d(img_channels, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        
        self.flatten = Flatten()
        self.fc1 = init_(nn.Linear(32*7*7, latent_size))


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return x


class FrameStacked(nn.Module):
    def __init__(self, net, n_stack, parallel=False, max_parallel=50):
        super().__init__()
        self.net = net
        self.n_stack = n_stack
        self.parallel = parallel
        self.max_parallel = max_parallel

    def _prepare_inputs(self, x):
        if isinstance(x, dict):
            x_dict = {}
            for k in x.keys():
                x_dict[k] = torch.chunk(x[k], self.n_stack, dim=1)
            xs = []
            for i in range(self.n_stack):
                dict_i = {k: v[i] for k, v in x_dict.items()}
                xs.append(dict_i)
        else:
            xs = torch.chunk(x, self.n_stack, dim=1)
        return xs

    def forward(self, x):
        xs = self._prepare_inputs(x)
        if self.parallel and len(x) <= self.max_parallel:  # not sure whats going on here
            xs = torch.cat(xs, dim=0)
            res = self.net(xs)
            res = torch.chunk(res, self.n_stack, dim=0)
        else:
            res = [self.net(x) for x in xs]
        out = torch.cat(res, dim=1)
        return out


class MemoryFrameStacked(FrameStacked):
    def __init__(self, net, n_stack, parallel=False, max_parallel=50, attrs_to_remember=[]):
        super().__init__(net, n_stack, parallel=parallel, max_parallel=max_parallel)
        self.attrs_to_remember = attrs_to_remember


    def forward(self, x, cache=None):
        xs = self._prepare_inputs(x)

        res = []
        if cache is not None:
            assert isinstance(cache, dict)
            for name in self.attrs_to_remember:
                if name not in cache:
                    cache[name] = []

        for x in xs:
            res.append(self.net(x))
            if cache is not None:
                # if cache is None, do not add anything. this fixes memory leak
                for name in self.attrs_to_remember:
                    cache[name].append(getattr(self.net, name))

        out = torch.cat(res, dim=1)
        return out

class FramestackResnet(nn.Module):
    def __init__(self, n_frames):
        super(FramestackResnet, self).__init__()
        self.n_frames = n_frames
        self.resnet = resnet18(pretrained=True)

    def forward(self, x):
        assert x.shape[1] / 3 == self.n_frames, "Dimensionality mismatch of input, is n_frames set right?"
        num_observations = x.shape[0]
        reshaped = x.reshape((x.shape[0] * self.n_frames, 3, x.shape[2], x.shape[3]))
        features = self.resnet(reshaped)
        return features.reshape((num_observations, features.shape[0] * features.shape[1] // num_observations))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class IgnoreInput(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.weights = Parameter(torch.Tensor(n_experts))
    
    def forward(self, x):
        sft = F.softmax(self.weights, dim=0)
        return torch.stack([sft for _ in range(x.shape[0])], dim=0)


class TaskonomyFeaturesOnlyNet(nn.Module):
    # Taskonomy features only, taskonomy encoder frozen
    def __init__(self, n_frames,
                 n_map_channels=0,
                 use_target=True,
                 output_size=512,
                 num_tasks=1,
                 extra_kwargs={},
                 ):
        super(TaskonomyFeaturesOnlyNet, self).__init__()
        self.n_frames = n_frames
        self.use_target = use_target
        self.use_map = n_map_channels > 0
        print(n_map_channels, self.use_map)
        self.map_channels = n_map_channels
        self.output_size = output_size

        if self.use_map:
            self.map_tower = atari_conv(self.n_frames * self.map_channels)
        if self.use_target:
            self.target_channels = 3
        else:
            self.target_channels = 0
        self.conv1 = nn.Conv2d(self.n_frames * (8 * num_tasks + self.target_channels), 32, 4, stride=2)
        self.flatten = Flatten()
        self.fc1 = init_(nn.Linear(32 * 7 * 7 * (self.use_map + 1), 1024))
        self.fc2 = init_(nn.Linear(1024, self.output_size))
        self.groupnorm = nn.GroupNorm(8, 8, affine=False)

        self.extra_kwargs = extra_kwargs
        self.features_scaling = 1
        if 'features_double' in extra_kwargs and extra_kwargs['features_double']:
            self.features_scaling = 2

        self.normalize_taskonomy = extra_kwargs['normalize_taskonomy'] if 'normalize_taskonomy' in extra_kwargs else True
        if not self.normalize_taskonomy:
            warnings.warn('Not normalizing taskonomy features in TaskonomyFeaturesOnlyNet, are you sure?')

    def forward(self, x, cache={}):
        try:
            x_taskonomy = x['taskonomy'] * self.features_scaling
        except AttributeError:
            x_taskonomy = x['taskonomy']
        try:
            if self.normalize_taskonomy:
                x_taskonomy = self.groupnorm(x_taskonomy)
        except AttributeError:
            x_taskonomy = self.groupnorm(x_taskonomy)
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

