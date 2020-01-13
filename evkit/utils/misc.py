import collections
import torch
import pprint
import string
from evkit.preprocess.transforms import rescale_centercrop_resize, rescale, grayscale_rescale, cross_modal_transform, \
    identity_transform, rescale_centercrop_resize_collated, map_pool_collated, map_pool, taskonomy_features_transform, \
    image_to_input_collated, taskonomy_multi_features_transform
from evkit.models.alexnet import alexnet_transform, alexnet_features_transform
from evkit.preprocess.baseline_transforms import blind, pixels_as_state
from evkit.models.srl_architectures import srl_features_transform
import warnings
remove_whitespace = str.maketrans('', '', string.whitespace)


def cfg_to_md(cfg, uuid):
    ''' Because tensorboard uses markdown'''
    return uuid + "\n\n    " + pprint.pformat((cfg)).replace("\n", "    \n").replace("\n \'", "\n    \'") + ""

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def is_interactive():
    try:
        ip = get_ipython()
        return ip.has_trait('kernel')
    except:
        return False

def is_cuda(model):
    return next(model.parameters()).is_cuda


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
        self._keys, self._vals = zip(*adict.items())
        self._keys, self._vals = list(self._keys), list(self._vals)

    def keys(self):
        return self._keys

    def vals(self):
        return self._vals


def compute_weight_norm(parameters):
    ''' no grads! '''
    total = 0.0
    count = 0
    for p in parameters:
        total += torch.sum(p.data**2)
        # total += p.numel()
        count += p.numel()
    return (total / count)

def get_number(name):
    """
    use regex to get the first integer in the name
    if none exists, return -1
    """
    try:
        num = int(re.findall("[0-9]+", name)[0])
    except:
        num = -1
    return num

def append_dict(d, u, stop_recurse_keys=[]):
    for k, v in u.items():
        if isinstance(v, collections.Mapping) and k not in stop_recurse_keys:
            d[k] = append_dict(d.get(k, {}), v, stop_recurse_keys=stop_recurse_keys)
        else:
            if k not in d:
                d[k] = []
            d[k].append(v)
    return d

def update_dict_deepcopy(d, u):  # we need a deep dictionary update
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update_dict_deepcopy(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def eval_dict_values(d):
    for k in d.keys():
        if isinstance(d[k], collections.Mapping):
            d[k] = eval_dict_values(d[k])
        elif isinstance(d[k], str):
            d[k] = eval(d[k].replace("---", "'"))
    return d


def search_and_replace_dict(model_kwargs, task_initial):
    for k, v in model_kwargs.items():
        if isinstance(v, collections.Mapping):
            search_and_replace_dict(v, task_initial)
        else:
            if isinstance(v, str) and 'encoder' in v and task_initial not in v:
                new_pth = v.replace('curvature', task_initial)  # TODO make this the string between / and encoder
                warnings.warn(f'BE CAREFUL - CHANGING ENCODER PATH: {v} is being replaced for {new_pth}')
                model_kwargs[k] = new_pth
    return
