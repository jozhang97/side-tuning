# Various ways to combine information from base and side(s)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_models import IdentityFn, ResidualLayer
from .model_utils import _make_layer

class MergeOperator(nn.Module):
    def __init__(self, dense, task_idx, dataset):
        super().__init__()
        self.dense = dense
        self.task_idx = task_idx
        self.dataset = dataset

    def __call__(self, base_encoding, side_encoding, additional_encodings=[])->torch.Tensor:
        pass

    @property
    def weights(self):
        return []

    @property
    def param(self):
        return -1

class BaseOnly(MergeOperator):
    def __call__(self, base_encoding, side_encoding, additional_encodings=[]):
        return base_encoding

class SideOnly(MergeOperator):
    def __call__(self, base_encoding, side_encoding, additional_encodings=[]):
        return side_encoding

class Summation(MergeOperator):
    def __call__(self, base_encoding, side_encoding, additional_encodings=[]):
        merged_encoding = base_encoding + side_encoding + sum(additional_encodings)
        return merged_encoding

class Product(MergeOperator):
    def __call__(self, base_encoding, side_encoding, additional_encodings=[]):
        merged_encoding = base_encoding * side_encoding
        for add_encoding in additional_encodings:
            merged_encoding *= add_encoding
        return merged_encoding


class Alpha(MergeOperator):
    def __init__(self, dense, task_idx, **kwargs):
        super().__init__(dense, task_idx, **kwargs)
        if dense:
            self.alphas = nn.Parameter(torch.tensor(0.0).repeat(task_idx + 2))  # 2 - one for base, one for current
        else:
            self.alphas = nn.Parameter(torch.tensor(0.0))

    @property
    def weights(self):
        if self.dense:
            weights = torch.softmax(self.alphas, dim=0)
        else:
            alpha_squashed = torch.sigmoid(self.alphas)
            weights = [alpha_squashed, 1 - alpha_squashed]
        return weights

    def __call__(self, base_encoding, side_encoding, additional_encodings=[]):
        # Note: returns alpha_squashed * base_encoding + (1 - alpha_squashed) * side_encoding when not dense
        outputs_to_merge = [base_encoding] + additional_encodings + [side_encoding]
        merged_encoding = torch.zeros_like(base_encoding) if isinstance(base_encoding, torch.Tensor) else torch.zeros_like(side_encoding)
        assert len(self.weights) == len(outputs_to_merge), f'# of outputs ({len(outputs_to_merge)}) != # of alphas ({len(weights)})'
        for a, out in zip(self.weights, outputs_to_merge):
            merged_encoding += a * out
        return merged_encoding

    @property
    def param(self):
        # :return:  weighting (alpha) on base network
        return self.weights[0].item()

class FiLMNet(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=1):
        super().__init__()
        if kernel_size == 3:
            net_kwargs = { 'kernel_size': 3, 'stride': 1, 'padding': 1 }
        elif kernel_size == 1:
            net_kwargs = { 'kernel_size': 1, 'stride': 1, 'padding': 0 }
        else:
            assert False, f'kernel size not recognized ({kernel_size})'

        self.base_layer = _make_layer(n_in, 64, **net_kwargs)
        self.mult_head = nn.Conv2d(64, n_out, bias=True, **net_kwargs)
        self.add_head = nn.Conv2d(64, n_out, bias=True, **net_kwargs)

    def forward(self, x):
        x1 = self.base_layer(x)
        mult_factor = self.mult_head(x1) + x
        add_factor = self.add_head(x1) + x
        return mult_factor, add_factor

class FiLM(MergeOperator):
    def __init__(self, dense, **kwargs):
        super().__init__(dense, **kwargs)
        assert not dense
        self.film = FiLMNet(n_in=8, n_out=8, kernel_size=1)

    def __call__(self, base_encoding, side_encoding, additional_encodings=[]):
        mult_factor, add_factor = self.film(side_encoding)
        merged_encoding = base_encoding * mult_factor + add_factor
        return merged_encoding

make_conv_layer = lambda: _make_layer(8, 8, num_groups=2, kernel_size=1, scaling=True, postlinear=True)
make_linear_layer = lambda: nn.Sequential(nn.Linear(64,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64,64))
# make_linear_layer = lambda: _make_layer(64, 64, kernel_size=1, scaling=True, postlinear=True, normalize=False, linear=True)

class MLP(MergeOperator):
    def __init__(self, dense, task_idx, dataset):
        super().__init__(dense, task_idx, dataset)

        if dataset == 'icifar':
            self.make_layer = make_linear_layer
        elif dataset == 'taskonomy':
            self.make_layer = make_conv_layer

        self.base_net = self.make_layer()
        self.side_net = IdentityFn()

        if dense:
            self.dense_side_nets = nn.ModuleList([self.make_layer() for _ in range(task_idx)])

    def __call__(self, base_encoding, side_encoding, additional_encodings=[]):
        merged_encoding = self.base_net(base_encoding) + self.side_net(side_encoding)
        if self.dense:
            merged_encoding += sum([net(add_encoding) for net, add_encoding in zip(self.dense_side_nets, additional_encodings)])
        return merged_encoding

class MLP2(MLP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.side_net = self.make_layer()

class ResMLP2(MLP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.side_net = ResidualLayer(self.make_layer())

class MLPHidden(MLP):
    def __call__(self, base_encoding, side_encoding, additional_encodings=[]):
        merged_encoding = self.base_net(base_encoding) + self.side_net(side_encoding)
        if self.dense:
            merged_encoding += sum([net(add_encoding) for net, add_encoding in zip(self.dense_side_nets, additional_encodings)])
        return F.ReLU(merged_encoding)
