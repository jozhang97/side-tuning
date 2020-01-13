import copy
import torch.nn as nn
import torch
from torchsummary import summary


class ChainedModel(nn.Module):

    def __init__(self, nets):
        super().__init__()
        self.nets = nets

    def forward(self, x):
        outputs = []
        for net in self.nets:
            x = net(x)
            outputs.append(x)
        return outputs


def ChainDirect(network_constructor, n_channels_list, universal_kwargs={}):
    print("Universal kwargs", network_constructor, universal_kwargs)
    return network_constructor(in_channels=n_channels_list[0],
                               out_channels=n_channels_list[1],
                               **universal_kwargs)


# class ChainDirect(nn.Module):

#     def __init__(self, network_constructor, n_channels_list, universal_kwargs={}):
#         super().__init__()
#         print("Universal kwargs", network_constructor,  universal_kwargs)
#         self.net = network_constructor(in_channels=n_channels_list[0],
#                                        out_channels=n_channels_list[1],
#                                        **universal_kwargs)

#     def initialize_from_checkpoints(self, checkpoint_paths, logger=None):
#         for i, (net, ckpt_fpath) in enumerate(zip([self.net], checkpoint_paths)):
#             if logger is not None:
#                 logger.info(f"Loading step {i} from {ckpt_fpath}")
#             checkpoint = torch.load(ckpt_fpath)
#             sd = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
#             net.load_state_dict(sd)
#         return self

#     def forward(self, x):
#         return self.net(x)

class ChainedEDModel(nn.Module):

    def __init__(self, network_constructor, n_channels_list, universal_kwargs={}):
        super().__init__()
        self.nets = nn.ModuleList()
        for i in range(len(n_channels_list) - 1):
            net = network_constructor(in_channels=n_channels_list[i],
                                      out_channels=n_channels_list[i + 1],
                                      **universal_kwargs)
            self.nets.append(net)

    def initialize_from_checkpoints(self, checkpoint_paths, logger=None):
        for i, (net, ckpt_fpath) in enumerate(zip(self.nets, checkpoint_paths)):
            if logger is not None:
                logger.info(f"Loading step {i} from {ckpt_fpath}")
            checkpoint = torch.load(ckpt_fpath)
            sd = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            net.load_state_dict(sd)
        return self

    def forward(self, x):
        outputs = []
        for net in self.nets:
            x = net(x)
            outputs.append(x)
        #             x = x[0]
        return outputs


# class ChainedEDModelHomo(nn.Module):

#     def __init__(self, network_constructor, n_channels_list, universal_kwargs={}):
#         super().__init__()
#         self.nets = nn.ModuleList()
#         for i in range(len(n_channels_list) - 1):
#             net = network_constructor(in_channels=n_channels_list[i],
#                                       out_channels=n_channels_list[i+1],
#                                       **universal_kwargs)
#             self.nets.append(net)

#     def initialize_from_checkpoints(self, checkpoint_paths, logger=None):
#         for i, (net, ckpt_fpath) in enumerate(zip(self.nets, checkpoint_paths)):
#             if logger is not None:
#                 logger.info(f"Loding step {i} from {ckpt_fpath}")
#             checkpoint = torch.load(ckpt_fpath)
#             sd = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
#             net.load_state_dict(sd)
#         return self

#     def forward(self, x):
#         outputs = []
#         for net in self.nets:
#             x = net(x)
#             outputs.append(x)
#         return outputs

class ChainedEDModelWithUncertaintyChannel(nn.Module):

    def __init__(self, network_constructor, n_channels_list, universal_kwargs={}):
        super().__init__()
        self.nets = nn.ModuleList()
        for i in range(len(n_channels_list) - 1):
            in_channels = n_channels_list[i] if i == 0 else n_channels_list[i] + 1
            net = network_constructor(in_channels=in_channels,
                                      out_channels=n_channels_list[i + 1],
                                      **universal_kwargs)
            self.nets.append(net)

    def __pad_with_zeros(self, weights):
        new_shape = list(weights.shape)
        new_shape[1] = new_shape[1] + 1
        new_params = torch.zeros(new_shape)
        new_params[:, :new_shape[1] - 1] = weights
        print(new_params.shape)
        return new_params

    def initialize_from_checkpoints(self, checkpoint_paths, logger=None):
        for i, (net, ckpt_fpath) in enumerate(zip(self.nets, checkpoint_paths)):
            if logger is not None:
                logger.info(f"Loding step {i} from {ckpt_fpath}")
            checkpoint = torch.load(ckpt_fpath)
            if logger is not None:
                logger.info(f"Loaded epoch {checkpoint['epoch']} from {ckpt_fpath}")
            sd = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            if i > 0:
                sd['down1.conv1.weight'] = self.__pad_with_zeros(sd['down1.conv1.weight'])
            net.load_state_dict(sd)
        return self

    def forward(self, x):
        outputs = []
        for net in self.nets:
            x = net(x)
            outputs.append(x)
            x = torch.cat(x, dim=1)
        return outputs


class ChainedEDModelWithLinearUncertaintyChannel(nn.Module):

    def __init__(self, network_constructor, n_channels_list, universal_kwargs={}):
        super().__init__()
        self.nets = nn.ModuleList()
        for i in range(len(n_channels_list) - 1):
            in_channels = n_channels_list[i] if i == 0 else 2 * n_channels_list[i]
            net = network_constructor(in_channels=in_channels,
                                      out_channels=n_channels_list[i + 1],
                                      **universal_kwargs)
            self.nets.append(net)

    def __pad_with_zeros(self, weights):
        new_shape = list(weights.shape)
        print(weights.shape)
        new_shape[1] = 2 * new_shape[1]
        new_params = torch.zeros(new_shape)
        new_params[:, :weights.shape[1]] = weights
        print(new_params.shape)
        return new_params

    def initialize_from_checkpoints(self, checkpoint_paths, logger=None):
        for i, (net, ckpt_fpath) in enumerate(zip(self.nets, checkpoint_paths)):
            if logger is not None:
                logger.info(f"Loding step {i} from {ckpt_fpath}")
            checkpoint = torch.load(ckpt_fpath)
            sd = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            if i > 0:
                sd['down1.conv1.weight'] = self.__pad_with_zeros(sd['down1.conv1.weight'])
            net.load_state_dict(sd)
        return self

    def forward(self, x):
        outputs = []
        for net in self.nets:
            x = net(x)
            outputs.append(x)
            x = torch.cat(x, dim=1)
        return outputs