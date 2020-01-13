import torch.nn as nn
import torch.nn.functional as F

from .student_models import FCN5, FCN4
from .model_utils import _make_layer
from tlkit.models.basic_models import EvalOnlyModel


class FCN5ProgressiveNoNewParam(FCN5):
    def __init__(self, early_fusion=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_fusion = early_fusion

    def forward(self, x, cache, task_idx:int=-1):
        cache, side_caches = cache
        x0 = x
        x = self.conv1(x)  # (64,64,64)
        x = x + cache[0]
        x = self.conv2(x)  # (256,32,32)
        x = x + cache[1]
        if self.use_residual:
            x = x + self.residual(x0)
        x2 = x
        x = self.conv3(x)  # (256,16,16)
        if not self.early_fusion:
            x = x + cache[2][:,:256,:,:]
        x = self.conv4(x)  # (64,16,16)
        if not self.early_fusion:
            x = x + cache[3][:,:64,:,:]
        x = self.conv5(x)
        x = x + self.skip(x2)
        if self.normalize_outputs:
            x = self.groupnorm(x)
        return x, []  # (8,16,16)

class FCN5Progressive(FCN5):
    def __init__(self, dense=False, task_idx=None, k=3, adapter='linear', extra_adapter=False, num_groups=2, *args, **kwargs):
        """
        :param dense: use states from all previous tasks (as opposed to just base task)
        :param task_idx: task number, used to determine how many lateral connections, only if dense
        :param k: kernel size (3x3) or (1x1)
        :param adapter: type of adapter to use (linear or mlp)
        :param extra_adapter: attach two adapters to the last hidden state - hopefully this makes this method stronger
        """
        super().__init__(*args, **kwargs)

        # Are we using 3x3 or 1x1 kernels?
        if k == 3:
            p = 1
        elif k == 1:
            p = 0
        else:
            assert False, f'do not know how to use kernel of size {k}'

        # Adapters from base network
        self.extra_adapter = extra_adapter
        if adapter == 'linear':
            self.adapter1 = nn.Conv2d(64, 64, kernel_size=k, stride=1, padding=p)
            self.adapter2 = nn.Conv2d(256, 256, kernel_size=k, stride=1, padding=p)
            self.adapter3 = nn.Conv2d(512, 256, kernel_size=k, stride=1, padding=p)
            self.adapter4 = nn.Conv2d(1024, 64, kernel_size=k, stride=1, padding=p)
            if self.extra_adapter:
                self.adapter5 = nn.Conv2d(2048, 64, kernel_size=k, stride=1, padding=p)
        elif adapter == 'mlp':
            self.adapter1 = _make_layer(64, 64, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
            self.adapter2 = _make_layer(256, 256, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
            self.adapter3 = _make_layer(512, 256, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
            self.adapter4 = _make_layer(1024, 64, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
            if self.extra_adapter:
                self.adapter5 = _make_layer(2048, 64, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
        else:
            assert False, f'Do not recognize pnn adapter {adapter}'

        # Adapters from side network
        self.task_idx_init = 0
        self.side_adapters1 = []
        self.side_adapters2 = []
        self.side_adapters3 = []
        self.side_adapters4 = []
        if dense:
            assert task_idx is not None, 'If dense connections from prev tasks, use side adapters!'
            self.task_idx_init = task_idx
            if adapter == 'linear':
                self.side_adapters1 = nn.ModuleList([nn.Conv2d(64, 64, kernel_size=1, stride=1) for _ in range(task_idx)])
                self.side_adapters2 = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=1, stride=1) for _ in range(task_idx)])
                self.side_adapters3 = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=1, stride=1) for _ in range(task_idx)])
                self.side_adapters4 = nn.ModuleList([nn.Conv2d(64, 64, kernel_size=1, stride=1) for _ in range(task_idx)])
            elif adapter == 'mlp':
                self.side_adapters1 = nn.ModuleList([_make_layer(64, 64, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True) for _ in range(task_idx)])
                self.side_adapters2 = nn.ModuleList([_make_layer(256, 256, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True) for _ in range(task_idx)])
                self.side_adapters3 = nn.ModuleList([_make_layer(256, 256, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True) for _ in range(task_idx)])
                self.side_adapters4 = nn.ModuleList([_make_layer(64, 64, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True) for _ in range(task_idx)])

    def forward(self, x, cache, task_idx:int=-1):
        """
        :param cache: [activations from base networks, (activations from other side networks)]
        :return: repr, activations from this side network
        """
        base_cache, side_caches = cache
        assert len(side_caches) == self.task_idx_init, f'Number of side caches ({len(side_caches)}) != Task ID ({task_idx})'
        x0 = x
        x_conv1 = self.conv1(x)  # (64,64,64)
        x_adapt1 = x_conv1 + self.adapter1(base_cache[0]) \
                   + sum([side_adapter1(side_cache[0]) for (side_adapter1, side_cache) in zip(self.side_adapters1, side_caches)])

        x_conv2 = self.conv2(x_adapt1)  # (256,32,32)
        x_adapt2 = x_conv2 + self.adapter2(base_cache[1]) \
                   + sum([side_adapter2(side_cache[1]) for (side_adapter2, side_cache) in zip(self.side_adapters2, side_caches)])

        if self.use_residual:
            x_adapt2 = x_adapt2 + self.residual(x0)
        x2 = x_adapt2
        x_conv3 = self.conv3(x_adapt2)  # (256,16,16)
        x_adapt3 = x_conv3 + self.adapter3(base_cache[2]) \
                   + sum([side_adapter3(side_cache[2]) for (side_adapter3, side_cache) in zip(self.side_adapters3, side_caches)])

        x_conv4 = self.conv4(x_adapt3)  # (64,16,16)
        x_adapt4 = x_conv4 + self.adapter4(base_cache[3]) \
                   + sum([side_adapter4(side_cache[3]) for (side_adapter4, side_cache) in zip(self.side_adapters4, side_caches)])

        if self.extra_adapter:
            x_adapt4 += self.adapter5(base_cache[4])

        x_conv5 = self.conv5(x_adapt4)
        x = x_conv5 + self.skip(x2)
        if self.normalize_outputs:
            x = self.groupnorm(x)
        return x, [x_adapt1.detach(), x_adapt2.detach(), x_adapt3.detach(), x_adapt4.detach()]  # (8,16,16)

class FCN5ProgressiveH(FCN5):
    def __init__(self, dense=False, task_idx=None, num_groups=2, img_channels=3, normalize_outputs=False, **kwargs):
        super().__init__(num_groups=num_groups, img_channels=img_channels, normalize_outputs=normalize_outputs, **kwargs)

        # Main FCN5s parameters
        self.conv1 = _make_layer(img_channels, 64, num_groups=num_groups, kernel_size=8, stride=4, padding=2)
        self.conv2 = _make_layer(64, 256, num_groups=num_groups, kernel_size=3, stride=2, padding=1)
        self.conv3 = _make_layer(256, 256, num_groups=num_groups, kernel_size=3, stride=2, padding=1)
        self.conv4 = _make_layer(256, 64, num_groups=num_groups, kernel_size=3, stride=1, padding=1)
        self.conv5 = _make_layer(64, 8, num_groups=num_groups, kernel_size=3, stride=1, padding=1)
        self.skip = _make_layer(256, 8, num_groups=num_groups, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        # Progressive Net adapters
        if dense:
            assert task_idx is not None, 'If dense connections from prev tasks, use side adapters!'
            self.task_idx_init = task_idx
            self.side_adapters1 = nn.ModuleList([_make_layer(64, 256, num_groups=num_groups, kernel_size=3, stride=2, padding=1, scaling=True, postlinear=True) for _ in range(task_idx)])
            self.side_adapters2 = nn.ModuleList([_make_layer(256, 256, num_groups=num_groups, kernel_size=3, stride=2, padding=1, scaling=True, postlinear=True) for _ in range(task_idx)])
            self.side_adapters3 = nn.ModuleList([_make_layer(256, 64, num_groups=num_groups, kernel_size=3, stride=1, padding=1, scaling=True, postlinear=True) for _ in range(task_idx)])
            self.side_adapters4 = nn.ModuleList([_make_layer(64, 8, num_groups=num_groups, kernel_size=3, stride=1, padding=1, scaling=True, postlinear=True) for _ in range(task_idx)])
        else:
            self.task_idx_init = 0
            self.side_adapters1 = []
            self.side_adapters2 = []
            self.side_adapters3 = []
            self.side_adapters4 = []

        self.adapter1 = _make_layer(256, 256, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
        self.adapter2 = _make_layer(512, 256, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
        self.adapter3 = _make_layer(1024, 64, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
        self.adapter4 = _make_layer(2048, 8, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)

        # Normalization
        self.normalize_outputs = normalize_outputs
        if normalize_outputs:
            self.groupnorm = nn.GroupNorm(num_groups, 8)


    def forward(self, x, cache, task_idx:int=-1):
        """
        :param cache: [activations from base networks, (activations from other side networks)]
        :return: repr, activations from this side network
        """
        base_cache, side_caches = cache
        assert len(side_caches) == self.task_idx_init, f'Number of side caches ({len(side_caches)}) != Task ID ({task_idx})'

        x_conv1 = self.conv1(x)  # (64,64,64)

        x_adapt1 = self.conv2[0](x_conv1) + self.adapter1(base_cache[1]) \
                   + sum([side_adapter1(side_cache[0]) for (side_adapter1, side_cache) in zip(self.side_adapters1, side_caches)])
        x_conv2 = self.relu(self.conv2[1](x_adapt1))  # (256,32,32)
        x2 = x_conv2

        x_adapt2 = self.conv3[0](x_conv2) + self.adapter2(base_cache[2]) \
                   + sum([side_adapter2(side_cache[1]) for (side_adapter2, side_cache) in zip(self.side_adapters2, side_caches)])
        x_conv3 = self.relu(self.conv3[1]((x_adapt2)))  # (256,16,16)

        x_adapt3 = self.conv4[0](x_conv3) + self.adapter3(base_cache[3]) \
                   + sum([side_adapter3(side_cache[2]) for (side_adapter3, side_cache) in zip(self.side_adapters3, side_caches)])
        x_conv4 = self.relu(self.conv4[1](x_adapt3))  # (64,16,16)

        x_adapt4 = self.conv5[0](x_conv4) + self.adapter4(base_cache[4]) \
                   + sum([side_adapter4(side_cache[3]) for (side_adapter4, side_cache) in zip(self.side_adapters4, side_caches)])
        x_conv5 = self.relu(self.conv5[1](x_adapt4))

        x = x_conv5 + self.skip(x2)
        if self.normalize_outputs:
            x = self.groupnorm(x)
        return x, [x_conv1.detach(), x_conv2.detach(), x_conv3.detach(), x_conv4.detach()]  # (8,16,16)


class FCN4Progressive(FCN4):
    def __init__(self, num_groups=2, img_channels=3, use_residual=False, normalize_outputs=False,
                 bsp=False, period=None, debug=False, projected=False, early_fusion=False, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = _make_layer(img_channels, 16, num_groups=num_groups, kernel_size=3, stride=1, padding=1, bsp=bsp, period=period, debug=debug, projected=projected)
        self.conv2 = _make_layer(16, 16, num_groups=num_groups, kernel_size=3, stride=2, padding=0, bsp=bsp, period=period, debug=debug, projected=projected)
        self.conv3 = _make_layer(16, 32, num_groups=num_groups, kernel_size=3, stride=2, bsp=bsp, period=period, debug=debug, projected=projected)
        self.conv5 = _make_layer(32, 64, num_groups=num_groups, kernel_size=3, stride=1, normalize=normalize_outputs, bsp=bsp, period=period, debug=debug, projected=projected)

        self.adapter1 = _make_layer(16, 16, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
        self.adapter2 = _make_layer(16, 16, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
        self.early_fusion = early_fusion
        if not self.early_fusion:
            self.adapter3 = _make_layer(32, 32, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)
            self.adapter4 = _make_layer(64, 64, num_groups=num_groups, kernel_size=1, scaling=True, postlinear=True)

    def forward(self, x, cache, task_idx:int=-1):
        """
        :param cache: [activations from base networks, (activations from other side networks)]
        :return: repr, activations from this side network
        """
        base_cache, side_caches = cache

        x = self.conv1(x) + self.adapter1(base_cache[0])  # (16,32,32)
        x = self.conv2(x) + F.max_pool2d(self.adapter2(base_cache[1]), kernel_size=2)[:,:,:15,:15]  # (16,15,15)
        x = self.conv3(x)
        if not self.early_fusion:
            x = x + F.max_pool2d(self.adapter3(base_cache[2]), kernel_size=2)[:,:,:7,:7]    # (32, 7, 7)
        x = self.conv5(x)[:,:,:4,:4]
        if not self.early_fusion:
            x = x + F.max_pool2d(self.adapter4(base_cache[3]), kernel_size=2) # (64, 5, 5)

        x = F.avg_pool2d(x, x.size()[3]).view(x.shape[0], 64)
        return x, []

class FCN4ProgressiveH(FCN4Progressive):
    def __init__(self, dense=False, task_idx=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = dense

        # Progressive Net adapters
        if dense:
            assert task_idx is not None, 'If dense connections from prev tasks, use side adapters!'
            self.task_idx_init = task_idx
            self.side_adapters1 = nn.ModuleList([_make_layer(16, 16, num_groups=2, kernel_size=3, stride=2, scaling=True, postlinear=True) for _ in range(task_idx)])
            self.side_adapters2 = nn.ModuleList([_make_layer(16, 32, num_groups=2, kernel_size=3, stride=2, scaling=True, postlinear=True) for _ in range(task_idx)])
            self.side_adapters3 = nn.ModuleList([_make_layer(32, 64, num_groups=2, kernel_size=3, stride=1, scaling=True, postlinear=True) for _ in range(task_idx)])
        else:
            self.task_idx_init = 0
            self.side_adapters1 = []
            self.side_adapters2 = []
            self.side_adapters3 = []
            self.side_adapters4 = []

    def forward(self, x, cache, task_idx:int=-1):
        """
        :param cache: [activations from base networks, (activations from other side networks)]
        :return: repr, activations from this side network
        """
        base_cache, side_caches = cache
        assert len(side_caches) == self.task_idx_init, f'Number of side caches ({len(side_caches)}) != Task ID ({task_idx})'

        x_adapt1 = self.conv1[0](x) + self.adapter1(base_cache[0])  # (16,32,32)
        x_l2 = F.relu(self.conv1[1](x_adapt1))
        x_adapt2 = self.conv2[0](x_l2) + F.max_pool2d(self.adapter2(base_cache[1]), kernel_size=2)[:,:,:15,:15] + \
                   sum([side_adapter1(side_cache[0]) for (side_adapter1, side_cache) in zip(self.side_adapters1, side_caches)])  # (16,15,15)
        x_l3 = F.relu(self.conv2[1](x_adapt2))
        x_adapt3 = self.conv3[0](x_l3) + F.max_pool2d(self.adapter3(base_cache[2]), kernel_size=2)[:,:,:7,:7] + \
                   sum([side_adapter2(side_cache[1]) for (side_adapter2, side_cache) in zip(self.side_adapters2, side_caches)])  # (32, 7, 7)
        x_l4 = F.relu(self.conv3[1](x_adapt3))
        x_adapt4 = self.conv5[0](x_l4)[:,:,:4,:4] + F.max_pool2d(self.adapter4(base_cache[3]), kernel_size=2) + \
                   sum([side_adapter3(side_cache[2])[:,:,:4,:4]  for (side_adapter3, side_cache) in zip(self.side_adapters3, side_caches)])  # (64, 5, 5)
        x_l5 = F.relu(self.conv5[1](x_adapt4))

        x_l5 = F.avg_pool2d(x_l5, x_l5.size()[3]).view(x_l5.shape[0], 64)
        return x_l5, [x_l2.detach(), x_l3.detach(), x_l4.detach()]
