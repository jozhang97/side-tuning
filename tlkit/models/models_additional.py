import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

from .model_utils import _make_layer, load_submodule
from .basic_models import zero_fn, ZeroFn, identity_fn
from tlkit.data.img_transforms import downsample_group_stack, RESCALE_0_1_NEG1_POS1

class SampleGroupStackModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SampleGroupStackModule, self).__init__()

    def forward(self, *args, **kwargs):
        return downsample_group_stack(*args, **kwargs)

    def requires_grad_(self, *args, **kwargs):
        pass


class ConstantModel():
    def __init__(self, data):
        if isinstance(data, str):
            if '.png' in data:
                img = Image.open(data)
                self.const = RESCALE_0_1_NEG1_POS1(transforms.ToTensor()(img))
            else:
                self.const = torch.load(data)
        else:
            self.const = data

    def forward(self, x):
        return self.const

    def to(self, device):
        self.const = self.const.to(device)

    def train(self, x):
        pass

    def __call__(self, x):
        return self.const


class EnsembleNet(nn.Module):
    def __init__(self, n_models, model_class, model_weights_path, **kwargs):
        super().__init__()
        self.nets = nn.ModuleList([load_submodule(eval(model_class), model_weights_path, kwargs) for _ in range(n_models)])

    def forward(self, x):
        return sum([net(x) for net in self.nets])


class BoostedNetwork(nn.Module):
    # T_i( E(x) + S_i(x).detach() + ... + S_{t-1}(x).detach() + S_t(x) )
    # encoder + transfer + decoder + [side_network_i, alpha_i for i in 1 to N]
    def __init__(self,
                 use_baked_encoding=False, normalize_pre_transfer=True,
                 encoder_class=None, encoder_weights_path=None, encoder_kwargs={},
                 transfer_network_class=None, transfer_network_weights_path=None, transfer_network_kwargs={},
                 sidetuner_network_class=None, sidetuner_network_weights_path=None, sidetuner_kwargs={},
                 decoder_class=None, decoder_weights_path=None, decoder_kwargs={},
                 tasks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                 ):
        super().__init__()
        self.encoder = load_submodule(eval(encoder_class), encoder_weights_path, encoder_kwargs, zero_fn)
        # self.side_network = load_submodule(sidetuner_network_class, sidetuner_network_weights_path, sidetuner_kwargs, zero_fn)
        self.side_network_args = (eval(sidetuner_network_class), sidetuner_network_weights_path, sidetuner_kwargs, ZeroFn())
        self.side_networks = nn.ModuleDict()

        self.transfer_network = load_submodule(eval(transfer_network_class), transfer_network_weights_path, transfer_network_kwargs, identity_fn)
        # self.transfer_network_args = (transfer_network_class, transfer_network_weights_path, transfer_network_kwargs, identity_fn)
        # self.transfer_networks = nn.ModuleDict()


        self.alphas = {} if sidetuner_network_class is None else nn.ParameterDict()

        for task_idx in tasks:
            task_id = str(task_idx)
            if task_id not in self.side_networks:
                self.side_networks[task_id] = load_submodule(*self.side_network_args)
                # self.transfer_networks[task_id] = load_submodule(*self.transfer_network_args)
                self.alphas[task_id] = nn.Parameter(torch.tensor(0.0))
        self.alphas['base'] = nn.Parameter(torch.tensor(0.0))
        assert decoder_class is None, 'we do not use decoder yet'
        self.decoder = load_submodule(eval(decoder_class), decoder_weights_path, decoder_kwargs, identity_fn)

        # self.alpha = nn.Parameter(torch.tensor(0.5))
        self.use_baked_encoding = use_baked_encoding
        self.normalize_pre_transfer = normalize_pre_transfer
        if self.normalize_pre_transfer:
            self.groupnorm = nn.GroupNorm(8, 8, affine=False)

    def forward(self, x, task_idx):
        task_id = str(task_idx)
        # print(task_id)
        if isinstance(x, dict):
            x_dict = x
            assert 'rgb_filled' in x_dict.keys(), 'need input images to work with'
            x = x_dict['rgb_filled']
            self.base_encoding = x_dict['taskonomy'] if 'taskonomy' in x_dict.keys() else self.encoder(x)
        else:
            if self.use_baked_encoding:
                x, self.base_encoding = x
            else:
                self.base_encoding = self.encoder(x)


        # if task_id not in self.side_networks:
        #     self.side_networks[task_id] = torch.nn.DataParallel(load_submodule(*self.side_network_args))
        #     self.transfer_networks[task_id] = torch.nn.DataParallel(load_submodule(*self.transfer_network_args))
        self.side_output = [self.base_encoding] + [self.side_networks[str(t)](x).detach() for t in range(task_idx)] + [self.side_networks[task_id](x)]
        # self.side_output =  [self.side_networks[str(t)](x).detach() for t in range(task_idx)] + [self.side_networks[task_id](x)]
        # self.side_network = self.side_networks[task_id]
        # self.transfer_network = self.transfer_networks[task_id]
        self.alpha = [self.alphas['base'].detach()] + [self.alphas[str(t)].detach() for t in range(task_idx)] + [self.alphas[task_id]]

        # self.side_output = self.side_network(x)
        alpha_squashed = torch.softmax(torch.tensor(self.alpha), dim=0)
        self.alpha = alpha_squashed[-1]
        self.merged_encoding = torch.zeros_like(self.side_output[-1])
        for a, out in zip(alpha_squashed, self.side_output):
            self.merged_encoding += a * out
        # self.merged_encoding = alpha_squashed * self.base_encoding + (1 - alpha_squashed) * self.side_output
        # self.merged_encoding = torch.sum(alpha_squashed * self.side_output, dim=-1)
        if self.normalize_pre_transfer:
            self.normalized_merged_encoding = self.groupnorm(self.merged_encoding)
            self.transfered_encoding = self.transfer_network(self.normalized_merged_encoding)
        else:
            self.transfered_encoding = self.transfer_network(self.merged_encoding)
        # return self.decoder(self.transfered_encoding)
        return self.transfered_encoding

    def transfer(self, x):
        return self.transfer_network(x)

    def decode(self, x):
        return self.decoder(x)

    def start_task(self, task_idx):
        pass

