import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import load_submodule
from .basic_models import identity_fn, zero_fn
from .student_models import FCN4Reshaped, FCN5
from .resnet_cifar import ResnetiCifar44NoLinear

from evkit.models.taskonomy_network import TaskonomyDecoder, TaskonomyEncoder

class GenericSidetuneNetwork(nn.Module):
    # D(T(E(x) + S(x)))
    def __init__(self, n_channels_in, n_channels_out,
                 use_baked_encoding=False, normalize_pre_transfer=True, add_base_encoding_post_transfer=False,
                 base_class=None, base_weights_path=None, base_kwargs={},
                 decoder_class=None, decoder_weights_path=None, decoder_kwargs={},
                 transfer_class=None, transfer_weights_path=None, transfer_kwargs={},
                 side_class=None, side_weights_path=None, side_kwargs={},
                 alpha_blend=True, alpha_kwargs={}, concat=False,
                 ):
        super().__init__()
        self.side = load_submodule(eval(str(side_class)), side_weights_path, side_kwargs, zero_fn)
        self.base = load_submodule(eval(str(base_class)), base_weights_path, base_kwargs, zero_fn)
        self.decoder = load_submodule(eval(str(decoder_class)), decoder_weights_path, decoder_kwargs, identity_fn)
        self.transfer = load_submodule(eval(str(transfer_class)), transfer_weights_path, transfer_kwargs, identity_fn)

        self.use_baked_encoding = use_baked_encoding
        self.add_base_encoding_post_transfer = add_base_encoding_post_transfer
        self.normalize_pre_transfer = normalize_pre_transfer
        if self.normalize_pre_transfer:
            self.groupnorm = nn.GroupNorm(8, 8, affine=False)

        self.alpha_blend = alpha_blend
        if self.alpha_blend:
            init_value = alpha_kwargs['init_value'] if 'init_value' in alpha_kwargs else 0.0
            self.alpha = nn.Parameter(torch.tensor(init_value))

        self.concat = concat
        assert not (self.alpha_blend and self.concat), 'Cannot merge with concat and alphablend together'

    def forward(self, x, task_idx=None, cache={}):
        if isinstance(x, dict):
            x_dict = x
            taskonomys = [k for k in x_dict.keys() if 'taskonomy' in k]
            assert 'rgb_filled' in x_dict.keys() or len(taskonomys) > 0, 'need image or taskonomy features'

            if 'rgb_filled' in x_dict.keys():
                x = x_dict['rgb_filled']

            if len(taskonomys) > 0:
                assert len(taskonomys) <= 1, 'not sure what to do with more than one taskonomy feature yet'
                self.base_encoding = x_dict[taskonomys[0]]
            else:
                try:
                    self.base_encoding = self.base(x, task_idx)
                except TypeError:
                    self.base_encoding = self.base(x)
        else:
            if self.use_baked_encoding:
                x, self.base_encoding = x
            else:
                try:
                    self.base_encoding = self.base(x, task_idx)
                except TypeError as e:
                    self.base_encoding = self.base(x)

                    # Forward Side Network
        try:
            self.side_output = self.side(x, task_idx)
        except TypeError as e:
            self.side_output = self.side(x)

        if isinstance(self.side_output, torch.Tensor) and isinstance(self.base_encoding, torch.Tensor):
            assert self.base_encoding.shape == self.side_output.shape, f'Shape of base and side are different. Base: {self.base_encoding.shape}, Side: {self.side_output.shape}'

        # Merge Side and Base Network
        if hasattr(self, 'alpha_blend') and self.alpha_blend:
            alpha_squashed = torch.sigmoid(self.alpha)
            self.merged_encoding = alpha_squashed * self.base_encoding + (1 - alpha_squashed) * self.side_output
        elif self.concat:
            assert isinstance(self.side_output, torch.Tensor) and isinstance(self.base_encoding, torch.Tensor), 'Cannot concat without base and side tensors'
            self.merged_encoding = torch.cat([self.base_encoding, self.side_output], dim=1)
        else:
            self.merged_encoding = 0.5 * self.base_encoding + 0.5 * self.side_output

        if self.normalize_pre_transfer:
            self.normalized_merged_encoding = self.groupnorm(self.merged_encoding)
            self.transfered_encoding = self.transfer(self.normalized_merged_encoding)
        else:
            self.transfered_encoding = self.transfer(self.merged_encoding)
        if self.add_base_encoding_post_transfer:
            self.transfered_encoding = self.transfered_encoding + self.base_encoding
        return self.decoder(self.transfered_encoding)

    def forward_transfer(self, x):
        return self.transfer(x)

    def forward_decode(self, x):
        return self.decoder(x)

    def start_training(self):
        if hasattr(self.base, 'start_training'):
            self.base.start_training()


class TransferConv3(nn.Module):
    def __init__(self, n_channels, n_channels_in=None, residual=False):
        # n_channels = n_channels_out
        super().__init__()
        if n_channels_in is None:
            n_channels_in = n_channels

        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.residual = residual
        self.n_channels = n_channels


    def forward(self, x):
        x_copy = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.residual:
            if x.shape != x_copy.shape:
                x_copy = x_copy[:,:self.n_channels,:,:]
            x = x + x_copy
        return x


class PreTransferedDecoder(nn.Module):
    def __init__(self,
                 transfer_class=None, transfer_weights_path=None, transfer_kwargs={},
                 decoder_class=None, decoder_weights_path=None, decoder_kwargs={}, **kwargs):
        super().__init__()
        self.transfer = load_submodule(eval(str(transfer_class)), transfer_weights_path, transfer_kwargs, identity_fn)
        self.decoder = load_submodule(eval(str(decoder_class)), decoder_weights_path, {**decoder_kwargs, **kwargs}, identity_fn)

    def forward(self, x):
        x = self.transfer(x)
        return self.decoder(x)

