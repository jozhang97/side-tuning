import torch.nn as nn

from .student_models import FCN5
from .model_utils import _make_layer, upsampler

class FCN5MidFeedback(FCN5):
    def __init__(self, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kernel_size == 3:
            net_kwargs = { 'kernel_size': 3, 'stride': 1, 'padding': 1 }
        elif kernel_size == 1:
            net_kwargs = { 'kernel_size': 1, 'stride': 1, 'padding': 0 }
        else:
            assert False, f'kernel size not recognized ({kernel_size})'
        self.fb_conv1 = _make_layer(8, 64, **net_kwargs)
        self.fb_conv2 = _make_layer(64, 8, **net_kwargs)
        self.feedback_net = nn.Sequential(self.fb_conv1, self.fb_conv2)

    def forward(self, x, task_idx:int=-1, cache={}):
        # Prepare feedback input
        last_repr = cache['last_repr']
        last_repr_tweeked = self.feedback_net(last_repr)
        last_repr_tweeked = upsampler(last_repr_tweeked)
        last_repr_tweeked = last_repr_tweeked.repeat(1,256//8,1,1)

        # Run forward
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + last_repr_tweeked
        x2 = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + self.skip(x2)

        if self.normalize_outputs:
            x = self.groupnorm(x)
        return last_repr + x

class FCN5LateFeedback(FCN5):
    # Late Feedback because the cache information is not incorporated in the base FCN5Skip
    # Instead, it is used later to augment the output
    # This does not have feedback. Output is not being used in input
    def __init__(self, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kernel_size == 3:
            net_kwargs = { 'kernel_size': 3, 'stride': 1, 'padding': 1 }
        elif kernel_size == 1:
            net_kwargs = { 'kernel_size': 1, 'stride': 1, 'padding': 0 }
        else:
            assert False, f'kernel size not recognized ({kernel_size})'
        self.fb_conv1 = _make_layer(8, 64, **net_kwargs)
        self.fb_conv2 = _make_layer(64, 8, **net_kwargs)
        self.feedback_net = nn.Sequential(self.fb_conv1, self.fb_conv2)

    def forward(self, x, task_idx:int=-1, cache={}):
        last_repr = cache['last_repr']

        ret_input_only = super().forward(x, task_idx)
        ret_output_only = self.feedback_net(last_repr)
        return last_repr + ret_input_only + ret_output_only
