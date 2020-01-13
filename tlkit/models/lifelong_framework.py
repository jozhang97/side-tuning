import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

from evkit.models.taskonomy_network import TaskonomyDecoder, TaskonomyEncoder, TaskonomyEncoderWithCache

import tlkit.models.merge_operators as merge_operators
from .basic_models import zero_fn, ZeroFn, identity_fn
from .fusion import FCN4ProgressiveH, FCN5ProgressiveH
from .model_utils import load_submodule, _make_layer
from .resnet_cifar import ResnetiCifar44NoLinearWithCache, ResnetiCifar44NoLinear
from .sidetune_architecture import GenericSidetuneNetwork, PreTransferedDecoder, TransferConv3
from .student_models import FCN5, FCN4Reshaped

class LifelongNetwork(nn.Module):
    def forward(self, x, task_idx=None):
        pass

    def start_training(self):
        pass

    def start_task(self, task_idx, train):
        pass

class LifelongSidetuneNetwork(LifelongNetwork):
    # T_i( R_i(B(x)) + S_i(x))
    # base + [side_i, alpha_i, transfer_i for i in 1 to N]
    def __init__(self,
         dataset='taskonomy',                    # cifar or taskonomy
         use_baked_encoding=False,               # Use saved encodings to speed up training
         normalize_pre_transfer=True,            # Normalize (GroupNorm) prior to applying transfer T_i(x)
         base_class=None,     base_weights_path=None,     base_kwargs={},      # Base setup
         transfer_class=None, transfer_weights_path=None, transfer_kwargs={},  # Transfer setup
         side_class=None,     side_weights_path=None,     side_kwargs={},      # side setup
         tasks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # Task list
         task_specific_transfer_kwargs=None,     # List of task-related kwargs for Transfer
         task_specific_side_kwargs=None,         # List of task-related kwargs for side
         dense=False,                            # Connections to all previous task side networks
         pnn=False,                              # Connections between lateral layers of base and side
         merge_method='merge_operators.Alpha',   # How to merge base and side
         base_uses_other_sensors=False,          # Base uses GT sensors
    ):
        super().__init__()
        if side_class is None:
            self.merge_method = 'merge_operators.BaseOnly'
        elif base_class is None:
            self.merge_method = 'merge_operators.SideOnly'
        else:
            self.merge_method = merge_method
        self.dataset = dataset
        self.tasks = tasks
        self.dense = dense
        self.pnn = pnn

        self.base = load_submodule(eval(str(base_class)), base_weights_path, base_kwargs, zero_fn)
        self.sides = nn.ModuleDict()
        self.transfers = nn.ModuleDict()
        self.merge_operators = nn.ModuleDict()

        task_specific_transfer_kwargs = [{} for _ in self.tasks] if task_specific_transfer_kwargs is None else task_specific_transfer_kwargs
        task_specific_side_kwargs = [{} for _ in self.tasks] if task_specific_side_kwargs is None else task_specific_side_kwargs

        if self.dense and self.pnn:
            for side_kwargs, task_idx in zip(task_specific_side_kwargs, tasks):
                side_kwargs['task_idx'] = task_idx
                side_kwargs['dense'] = True

        # Set up each task
        for task_idx in self.tasks:
            task_id = str(task_idx)
            if task_id in self.sides:
                continue
            merged_side_kwargs = {**side_kwargs, **task_specific_side_kwargs[task_idx]}
            merged_transfer_kwargs = {**transfer_kwargs, **task_specific_transfer_kwargs[task_idx]}

            self.sides[task_id] = load_submodule(eval(str(side_class)), side_weights_path, merged_side_kwargs, ZeroFn())
            self.transfers[task_id] = load_submodule(eval(str(transfer_class)), transfer_weights_path, merged_transfer_kwargs, identity_fn)
            self.merge_operators[task_id] = eval(self.merge_method)(dense=self.dense, task_idx=task_idx, dataset=self.dataset)

        self.use_baked_encoding = use_baked_encoding
        self.base_uses_other_sensors = base_uses_other_sensors
        self.normalize_pre_transfer = normalize_pre_transfer
        if self.normalize_pre_transfer:
            self.groupnorm = nn.GroupNorm(8, 8, affine=False)
        self.cache = {}  # additional input to side network
        self.eval()  # Always initialize model in eval mode

    def forward(self, x, task_idx=None, pass_i=0):
        if task_idx is None:
            warnings.warn('No task_idx is passed, are you sure? (only should do this for torchsummary)')
            task_idx = 0
        task_id = str(task_idx)

        # Forward base network
        if pass_i == 0:
            self.base_encoding = self.forward_base(x, task_idx)
            self.cache = {'last_repr': self.base_encoding}

        # Get task specific networks
        self.side = self.sides[task_id]
        self.transfer = self.transfers[task_id]
        self.merge_operator = self.merge_operators[task_id]

        if self.pnn:
            # Split base encoding from cache
            assert isinstance(self.base, TaskonomyEncoderWithCache) or isinstance(self.base, ResnetiCifar44NoLinearWithCache), 'PNN needs to have cache!'
            self.base_encoding, pnn_base_cache = self.base_encoding
            pnn_side_caches = []
            prev_side_encodings = []
            pnn_full_cache = [pnn_base_cache, pnn_side_caches]

            # Populate PNN side networks cache
            if self.dense:
                for t in range(task_idx):
                    this_side_encoding, this_side_cache = self.sides[str(t)](x, pnn_full_cache)
                    pnn_side_caches.append(this_side_cache)
                    prev_side_encodings.append(this_side_encoding.detach())

        # Forward side network
        if self.pnn:
            self.side_encoding, _ = self.side(x, pnn_full_cache)
        else:
            self.side_encoding = self.side(x, cache=self.cache)
            self.cache['last_repr'] = self.side_encoding

        # Merge base and side(s)
        additional_encodings = []
        if self.dense and self.pnn:
            additional_encodings = prev_side_encodings
        elif self.dense:
            additional_encodings = [self.sides[str(t)](x).detach() for t in range(task_idx)]
        self.merged_encoding = self.merge_operator(self.base_encoding, self.side_encoding, additional_encodings)

        # Maybe normalize then Transfer
        if self.normalize_pre_transfer:
            self.normalized_merged_encoding = self.groupnorm(self.merged_encoding)
            self.transfered_encoding = self.transfer(self.normalized_merged_encoding)
        else:
            self.transfered_encoding = self.transfer(self.merged_encoding)

        return self.transfered_encoding

    def forward_base(self, x, task_idx):
        # Use baked encoding if possible
        if isinstance(x, dict):
            x_dict = x
            assert 'rgb_filled' in x_dict.keys(), 'need input images to work with'
            x = x_dict['rgb_filled']
            if 'taskonomy' in x_dict:
                base_encoding = x_dict['taskonomy']
            else:
                try:
                    base_encoding = self.base(x, task_idx)
                except TypeError:
                    base_encoding = self.base(x)
        else:
            if self.use_baked_encoding:
                x, base_encoding = x
            else:
                if self.base_uses_other_sensors:
                    assert isinstance(x, list) and len(x) > 1, 'Must have additional sensors for base!'
                    x, other_sensors = x[0], x[1:]

                    assert len(other_sensors) == 1, 'Our system can only take ONE other_sensors'
                    other_sensors = other_sensors[0]
                    base_input = other_sensors
                else:
                    base_input = x

                try:
                    base_encoding = self.base(base_input, task_idx)  # model should be able to support with and without psp
                except TypeError as e:
                    base_encoding = self.base(base_input)  # these are old models who have not added support for psp
        return base_encoding

    def forward_transfer(self, x):
        return self.transfer(x)

    def start_task(self, task_idx, train, print_alpha=False) -> list:
        # For efficiency, turn off task-specific parameters when not using
        for task in self.tasks:
            task_id = str(task)
            if task == task_idx or (task < task_idx and self.dense):
                self.sides[task_id].cuda()
                self.transfers[task_id].cuda()
                self.merge_operators[task_id].cuda()

                self.sides[task_id].requires_grad_(train)
                self.transfers[task_id].requires_grad_(train)
                self.merge_operators[task_id].requires_grad_(train)
            else:
                self.sides[task_id].cpu()
                self.transfers[task_id].cpu()
                self.merge_operators[task_id].cpu()

                self.sides[task_id].requires_grad_(False)
                self.transfers[task_id].requires_grad_(False)
                self.merge_operators[task_id].requires_grad_(False)

                self._set_grad_to_none(self.sides[task_id])
                self._set_grad_to_none(self.transfers[task_id])
                self._set_grad_to_none(self.merge_operators[task_id])

        if print_alpha:
            alphas = [self.merge_operators[str(task_idx)].param for task_idx in self.tasks]
            print(f'Setting grad True for task {task_idx} and others False. \n\t Alphas: {alphas}')
        return [p for p in self.parameters() if p.requires_grad]

    def _set_grad_to_none(self, vars):
        # seems a bit hacky but important to stop Adam from updating
        if isinstance(vars, nn.parameter.Parameter):
            vars.grad = None
        elif isinstance(vars, nn.Module):
            for p in vars.parameters():
                p.grad = None

    def start_training(self):
        if hasattr(self.base, 'start_training'):
            self.base.start_training()
