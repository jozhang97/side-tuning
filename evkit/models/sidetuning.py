import torch
import torch.nn as nn

from evkit.models.architectures import MemoryFrameStacked, TaskonomyFeaturesOnlyNet
from evkit.utils.misc import count_total_parameters, count_trainable_parameters
from tlkit.models.sidetune_architecture import GenericSidetuneNetwork
import warnings

class RLSidetuneWrapper(nn.Module):
    # Taskonomy features only, taskonomy encoder frozen
    # wraps any perception model, replacing x['taskonomy'] with x['taskonomy'] + sidetune(x)
    def __init__(self, n_frames, blind=False, **kwargs):
        super(RLSidetuneWrapper, self).__init__()

        extra_kwargs = kwargs.pop('extra_kwargs')  # extra_kwargs are used just for this wrapper
        assert 'main_perception_network' in extra_kwargs, 'For RLSidetuneWrapper, need to include main class'
        assert 'sidetune_kwargs' in extra_kwargs, 'Cannot use sidetune network without kwargs'

        self.main_perception = eval(extra_kwargs['main_perception_network'])(n_frames, **kwargs)

        self.sidetuner = GenericSidetuneNetwork(**extra_kwargs['sidetune_kwargs'])
        attrs_to_remember = extra_kwargs['attrs_to_remember'] if 'attrs_to_remember' in extra_kwargs else []
        self.sidetuner = MemoryFrameStacked(self.sidetuner, n_frames, attrs_to_remember=attrs_to_remember)

        self.blind = blind
        assert not (self.blind and count_trainable_parameters(self.sidetuner) > 5), \
            'Cannot be blind and have many sidetuner parameters'

        self.n_frames = n_frames

    def forward(self, x, cache=None):
        # run sidetune network and populate cache
        if hasattr(self, 'blind') and self.blind:
            sample = x[list(x.keys())[0]]
            batch_size = sample.shape[0]
            x_sidetune = torch.zeros((batch_size, 8 * self.n_frames, 16, 16)).to(sample.device)
        else:
            x_sidetune = self.sidetuner(x, cache)  # taskonomy + F(rgb_filled)
        contains_taskonomy = 'taskonomy' in x
        if contains_taskonomy:
            taskonomy_copy = x['taskonomy']
        x['taskonomy'] = x_sidetune
        try:
            ret = self.main_perception(x, cache)
        except TypeError:
            ret = self.main_perception(x)
        if contains_taskonomy:
            x['taskonomy'] = taskonomy_copy
        else:
            # if there was no taskonomy key originally and it is later added,
            # the element will be used in future updates incorrectly
            del x['taskonomy']
        return ret


# depreciated, use model agnostic wrapper RLSidetuneWrapper
class RLSidetuneNetwork(nn.Module):
    # Taskonomy features only, taskonomy encoder frozen
    def __init__(self, n_frames,
                 n_map_channels=0,
                 use_target=True,
                 output_size=512,
                 num_tasks=1, extra_kwargs={}):
        super(RLSidetuneNetwork, self).__init__()
        assert 'sidetune_kwargs' in extra_kwargs, 'Cannot use sidetune network without kwargs'
        self.sidetuner = GenericSidetuneNetwork(**extra_kwargs['sidetune_kwargs'])
        attrs_to_remember = extra_kwargs['attrs_to_remember'] if 'attrs_to_remember' in extra_kwargs else []
        self.sidetuner = MemoryFrameStacked(self.sidetuner, n_frames, attrs_to_remember=attrs_to_remember)
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
        self.conv1 = nn.Conv2d(self.n_frames * (8 * num_tasks + self.target_channels), 32, 4, stride=2)
        self.flatten = Flatten()
        self.fc1 = init_(nn.Linear(32 * 7 * 7 * (self.use_map + 1), 1024))
        self.fc2 = init_(nn.Linear(1024, self.output_size))
        self.groupnorm = nn.GroupNorm(8, 8, affine=False)

    def forward(self, x, cache):
        # run sidetune network and populate cache
        x_sidetune = self.sidetuner(x, cache)  # taskonomy + F(rgb_filled)
        x_sidetune = self.groupnorm(x_sidetune)
        if self.use_target:
            x_sidetune = torch.cat([x_sidetune, x["target"]], dim=1)
        x_sidetune = F.relu(self.conv1(x_sidetune))
        if self.use_map:
            x_map = x['map']
            x_map = self.map_tower(x_map)
            x_sidetune = torch.cat([x_map, x_sidetune], dim=1)
        x = self.flatten(x_sidetune)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


