import torch.nn as nn
from torch.nn.modules.upsampling import Upsample

from tlkit.utils import load_state_dict_from_path
from .superposition import HashConv2d, ProjectedConv2d
from .basic_models import zero_fn, ScaleLayer

upsampler = Upsample(scale_factor=2, mode='nearest')

def load_submodule(model_class, model_weights_path, model_kwargs, backup_fn=zero_fn):
    # If there is a model, use it! If there is initialization, use it! If neither, use backup_fn
    if model_class is not None:
        model = model_class(**model_kwargs)
        if model_weights_path is not None:
            model, _ = load_state_dict_from_path(model, model_weights_path)
    else:
        model = backup_fn
        assert model_weights_path is None, 'cannot have weights without model'
    return model

def _make_layer(in_channels, out_channels, num_groups=2, kernel_size=3, stride=1, padding=0, dilation=1, normalize=True,
                bsp=False, period=None, debug=False, projected=False, scaling=False, postlinear=False, linear=False):
    assert not (bsp and projected), 'cannot do bsp and projectedconv'
    if linear:
        conv = nn.Linear(in_channels, out_channels, bias=False)
    elif bsp:
        assert dilation == 1, 'Dilation is not implemented for binary superposition'
        assert period is not None, 'Need to specify period'
        conv = HashConv2d(in_channels, out_channels, kernel_size=kernel_size, period=period, stride=stride, padding=padding, bias=False, debug=debug)
    elif projected:
        assert dilation == 1, 'Dilation is not implemented for projected conv'
        conv = ProjectedConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation)
    gn = nn.GroupNorm(num_groups, out_channels)
    relu = nn.ReLU()

    layers = [conv, relu]
    if normalize:
        layers = [conv, gn, relu]
    if scaling:
        layers = [ScaleLayer(.9)] + layers
    if postlinear:
        if linear:
            layers = layers + [nn.Linear(in_channels, out_channels)]
        else:
            layers = layers + [nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)]
    return nn.Sequential(*layers)

