from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision


from evkit.models.architectures import FrameStacked, Flatten, atari_conv, init_, TaskonomyFeaturesOnlyNet  # important for old ckpts
from evkit.rl.utils import init, init_normc_


_all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, normalize_outputs=False, eval_only=True, train=False, stop_layer=None):
        super(AlexNet, self).__init__()
        assert normalize_outputs == False, 'AlexNet cannot set normalize_outputs to True'
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.stop_layer = stop_layer
        self.eval_only = eval_only
        if self.eval_only:
            self.eval()


    def forward(self, x, stop_layer=19, cache={}):
        if self.stop_layer is not None:
            stop_layer = self.stop_layer  # overwrite input parameter!
        if stop_layer == 'conv5':
            stop_layer = 11
        elif stop_layer == 'fc7':
            stop_layer = 19
        featuremodulelist = list(self.features.modules())[1:]
        classifiermodulelist = list(self.classifier.modules())[1:]
        if stop_layer == None:
            stop_layer = len(featuremodulelist) + 1 + len(classifiermodulelist)
        for i, l in enumerate(featuremodulelist[:stop_layer]):
            x = l(x)
        if stop_layer <= len(featuremodulelist) - 1:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        if stop_layer ==  len(featuremodulelist):
            return x
        for i, l in enumerate(classifiermodulelist[:stop_layer - (len(featuremodulelist) + 1)]):
            x = l(x)
        return x
    def train(self, val):
        if val and self.eval_only:
            warnings.warn("Ignoring 'train()' in TaskonomyEncoder since 'eval_only' was set during initialization.", RuntimeWarning)
        else:
            return super().train(val)


def alexnet(pretrained=False, load_path=None, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        if load_path is None:
            model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        else: 
            model.load_state_dict(torch.load(load_path))
    return model
    

    
    


def alexnet_transform(output_size, dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)
            
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    def _rescale_centercrop_resize_thunk(obs_space):
        obs_shape = obs_space.shape
        obs_min_wh = min(obs_shape[:2])
        output_wh = output_size[-2:]  # The out
        processed_env_shape = output_size
        
        pipeline1 = vision.transforms.Compose([
                vision.transforms.ToPILImage(),
                vision.transforms.CenterCrop([obs_min_wh, obs_min_wh]),
                vision.transforms.Resize(output_wh),    
                vision.transforms.ToTensor(),
                vision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ])

        def pipeline(im):
            x = pipeline1(im)
            return x
        
        return pipeline, spaces.Box(-1, 1, output_size, dtype)
    return _rescale_centercrop_resize_thunk


def alexnet_features_transform(task_path, dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)
            
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
#     _rescale_thunk = alexnet_transform((3, 224, 224))
    net = alexnet(pretrained=True, load_path=task_path).cuda()
    net.eval()

    def encode(x):
        if task_path == 'pixels_as_state':
            return x
        with torch.no_grad():
            return net(x, 'conv5')
    
    def _taskonomy_features_transform_thunk(obs_space):
        def pipeline(x):
            x = torch.Tensor(x).cuda()
            x = encode(x)
            return x.cpu()
        if task_path == 'pixels_as_state':
            raise NotImplementedError('pixels_as_state not defined for alexnet transform')
        else:
            return pipeline, spaces.Box(-1, 1, (256, 13, 13), dtype)
    
    return _taskonomy_features_transform_thunk



class AlexNetFeaturesOnlyNet(nn.Module):
    # Taskonomy features only, taskonomy encoder frozen
    def __init__(self, n_frames, n_map_channels=0, use_target=True,
                 output_size=512, extra_kwargs={}):
        super(AlexNetFeaturesOnlyNet, self).__init__()
        self.n_frames = n_frames
        self.use_target = use_target
        self.use_map = n_map_channels > 0
        self.map_channels = n_map_channels
        self.output_size = output_size

        if self.use_map:
            self.map_tower = atari_conv(self.n_frames * self.map_channels)
        if self.use_target:
            self.target_channels = 3
        else:
            self.target_channels = 0
        self.conv1 = nn.Conv2d(self.n_frames * (256 + self.target_channels), 64, 4, stride=2)
        self.flatten = Flatten()
        self.fc1 = init_(nn.Linear(32 * 7 * 7 * (self.use_map) + 64 * 5 * 5 * 1, 1024))
        self.fc2 = init_(nn.Linear(1024, self.output_size))
        self.groupnorm = nn.GroupNorm(8, 8, affine=False)

    def forward(self, x, cache={}):
        x_taskonomy = x['taskonomy']
        x_taskonomy = self.groupnorm(x_taskonomy)
        if self.use_target:
            x_taskonomy = torch.cat([x_taskonomy, x["target"]], dim=1)
        x_taskonomy = F.relu(self.conv1(x_taskonomy))
        if self.use_map:
            x_map = x['map']
            x_map = self.map_tower(x_map)
#             x_taskonomy = torch.cat([x_map, x_taskonomy], dim=1)
            x_taskonomy = torch.cat([self.flatten(x_taskonomy), self.flatten(x_map)], dim=1)
        x = self.flatten(x_taskonomy)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    

