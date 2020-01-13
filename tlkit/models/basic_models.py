import warnings

import torch
import torch.nn as nn

# Because we cannot pickle lambda functions
class IdentityFn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x

    def requires_grad_(self, *args, **kwargs):
        pass

def identity_fn(x):
    return x

class ZeroFn(nn.Module):
    def forward(self, *args, **kwargs):
        return 0.0

    def requires_grad_(self, *args, **kwargs):
        pass

def zero_fn(x):
    return 0.0

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResidualLayer(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return x + self.net(x)

class EvalOnlyModel(nn.Module):
    def __init__(self, eval_only=None, train=False, **kwargs):
        super().__init__()
        if eval_only is None:
            warnings.warn(f'Model eval_only flag is not set for {type(self)}. Defaulting to True')
            eval_only = True

        if train:
            warnings.warn('Model train flag is deprecated')

        self.eval_only = eval_only


    def forward(self, x, cache={}, time_idx:int=-1):
        pass

    def train(self, train):
        if self.eval_only:
            super().train(False)
            for p in self.parameters():  # This must be done after parameters are initialized
                p.requires_grad = False

        if train and self.eval_only:
            warnings.warn("Ignoring 'train()' in TaskonomyEncoder since 'eval_only' was set during initialization.", RuntimeWarning)
        else:
            return super().train(train)