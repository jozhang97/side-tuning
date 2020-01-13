from tlkit.utils import forward_sequential
import torch.nn as nn
import torch.nn.functional as F

from tlkit.models.model_utils import _make_layer
from tlkit.models.basic_models import EvalOnlyModel


class FCN5(EvalOnlyModel):
	def __init__(self, num_groups=2, img_channels=3, use_residual=False, normalize_outputs=False,
				 	   bsp=False, period=None, projected=False, final_act=True, **kwargs):
		super(FCN5, self).__init__(**kwargs)
		self.conv1 = _make_layer(img_channels, 64, num_groups=num_groups, kernel_size=8, stride=4, padding=2, bsp=bsp, period=period, projected=projected)
		self.conv2 = _make_layer(64, 256, num_groups=num_groups, kernel_size=3, stride=2, padding=1, bsp=bsp, period=period, projected=projected)

		self.conv3 = _make_layer(256, 256, num_groups=num_groups, kernel_size=3, stride=2, padding=1, bsp=bsp, period=period, projected=projected)
		self.conv4 = _make_layer(256, 64, num_groups=num_groups, kernel_size=3, stride=1, padding=1, bsp=bsp, period=period, projected=projected)
		self.conv5 = _make_layer(64, 8, num_groups=num_groups, kernel_size=3, stride=1, padding=1, bsp=bsp, period=period, projected=projected)

		self.skip = _make_layer(256, 8, num_groups=num_groups, kernel_size=3, stride=2, padding=1, bsp=bsp, period=period, projected=projected)

		self.normalize_outputs = normalize_outputs
		self.final_act = final_act
		if normalize_outputs:
			self.groupnorm = nn.GroupNorm(num_groups, 8)

		self.use_residual = use_residual
		if use_residual:
			res1 = nn.Conv2d(img_channels, 64, kernel_size=8, stride=4, padding=2, bias=False, dilation=1)
			res2 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False, dilation=1)
			# res2 = _make_layer(64, 256, num_groups=num_groups, kernel_size=3, stride=2, padding=1, normalize=False)
			self.residual = nn.Sequential(res1, res2)

		self.bsp = bsp

	def forward(self, x, task_idx:int=-1, cache={}):
		x0 = x
		if self.bsp:
			x = forward_sequential(x, self.conv1, task_idx)
			x = forward_sequential(x, self.conv2, task_idx)
		else:
			x = self.conv1(x)
			x = self.conv2(x)
		if self.use_residual:
			x = x + self.residual(x0)
		x2 = x
		if self.bsp:
			x = forward_sequential(x, self.conv3, task_idx)
			x = forward_sequential(x, self.conv4, task_idx)
			x = forward_sequential(x, self.conv5, task_idx)
			x = x + forward_sequential(x2, self.skip, task_idx)
		else:
			x = self.conv3(x)
			x = self.conv4(x)
			if self.final_act:
				x = self.conv5(x)
			else:
				x = self.conv5[0](x)
			x = x + self.skip(x2)
			# x = self.conv5(x)
			# if self.final_act:
			# 	x = x + self.skip(x2)
			# else:
			# 	x = x + self.skip[0](x2)

		if self.normalize_outputs:
			x = self.groupnorm(x)
		return x

class FCN8(EvalOnlyModel):
	# Total params: 3,361,440 so more than sidetune on taskonomy 12
	def __init__(self, img_channels=3, normalize_outputs=False, **kwargs):
		super(FCN8, self).__init__(**kwargs)
		self.conv1 = _make_layer(img_channels, 64, kernel_size=8, stride=4, padding=2)
		self.conv2 = _make_layer(64, 128, kernel_size=3, stride=2, padding=1)
		self.conv3 = _make_layer(128, 256, kernel_size=3, stride=2, padding=1)
		self.conv4 = _make_layer(256, 256, kernel_size=3, stride=1, padding=1)
		self.conv5 = _make_layer(256, 256, kernel_size=3, stride=1, padding=1)
		self.conv6 = _make_layer(256, 256, kernel_size=3, stride=1, padding=1)
		self.conv7 = _make_layer(256, 128, kernel_size=3, stride=1, padding=1)
		self.conv8 = _make_layer(128, 8, kernel_size=3, stride=1, padding=1)

		self.skip1 = _make_layer(128, 256, kernel_size=3, stride=2, padding=1)
		self.skip2 = _make_layer(256, 256, kernel_size=3, stride=1, padding=1)
		self.skip3 = _make_layer(256, 8, kernel_size=3, stride=1, padding=1)

		self.normalize_outputs = normalize_outputs
		if self.normalize_outputs:
			self.groupnorm = nn.GroupNorm(2, 8)

	def forward(self, x, task_idx:int=-1, cache={}):
		x = self.conv1(x)
		x = self.conv2(x)
		x2 = x

		x = self.conv3(x)
		x = self.conv4(x)
		x = x + self.skip1(x2)
		x4 = x

		x = self.conv5(x)
		x = self.conv6(x)
		x = x + self.skip2(x4)
		x6 = x

		x = self.conv7(x)
		x = self.conv8(x)
		x = x + self.skip3(x6)

		if self.normalize_outputs:
			x = self.groupnorm(x)
		return x

class FCN4(EvalOnlyModel):
	def __init__(self, num_groups=2, img_channels=3, use_residual=False, normalize_outputs=False,
				 bsp=False, period=None, debug=False, projected=False, final_act=True, **kwargs):
		super(FCN4, self).__init__(**kwargs)
		self.conv1 = _make_layer(img_channels, 16, num_groups=num_groups, kernel_size=3, stride=1, padding=1, bsp=bsp, period=period, debug=debug, projected=projected)
		self.conv2 = _make_layer(16, 16, num_groups=num_groups, kernel_size=3, stride=2, padding=0, bsp=bsp, period=period, debug=debug, projected=projected)
		self.conv3 = _make_layer(16, 32, num_groups=num_groups, kernel_size=3, stride=2, bsp=bsp, period=period, debug=debug, projected=projected)
		self.conv5 = _make_layer(32, 64, num_groups=num_groups, kernel_size=3, stride=1, normalize=normalize_outputs, bsp=bsp, period=period, debug=debug, projected=projected)

		self.bsp = bsp
		self.use_residual = use_residual
		self.final_act = final_act
		if use_residual:
			res1 = nn.Conv2d(img_channels, 8, kernel_size=3, stride=1, padding=0, bias=False, dilation=2)
			res2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=0, bias=False, dilation=2)
			res3 = nn.Conv2d(8, 64, kernel_size=3, stride=2, bias=False, dilation=1)
			self.residual = nn.Sequential(res1, res2, res3)

	def forward(self, x, task_idx:int=-1):
		if self.bsp:
			x = forward_sequential(x, self.conv1, task_idx)
			x = forward_sequential(x, self.conv2, task_idx)
			x = forward_sequential(x, self.conv3, task_idx)
			x = forward_sequential(x, self.conv5, task_idx)
		else:
			x = self.conv1(x)
			x = self.conv2(x)
			x = self.conv3(x)
			if self.final_act:
				x = self.conv5(x)
			else:
				x = self.conv5[0](x)
		if self.use_residual:
			res = self.residual(x)
			x = x + res
		return x

class FCN4Reshaped(FCN4):
	def forward(self, x, cache={}, time_idx:int=-1):
		x = super().forward(x, time_idx)
		x = F.avg_pool2d(x, x.size()[3]).view(x.shape[0], 64)
		return x

class FCN3(EvalOnlyModel):
	def __init__(self, num_groups=2, img_channels=3, normalize_outputs=False, **kwargs):
		super(FCN3, self).__init__(**kwargs)
		self.conv1 = _make_layer(img_channels, 64, num_groups=num_groups, kernel_size=8, stride=4, padding=1)
		self.conv2 = _make_layer(64, 256, num_groups=num_groups, kernel_size=3, stride=2, padding=2)
		self.conv3 = _make_layer(256, 8, num_groups=num_groups, kernel_size=3, stride=2, normalize=normalize_outputs)

	def forward(self, x, task_idx:int=-1, cache={}):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		return x

FCN5Skip = FCN5

if __name__ == '__main__':
	from torchsummary import summary
	net = FCN8()
	try:
		net = net.cuda()
	except:
		pass

	summary(net, (3, 224, 224))

