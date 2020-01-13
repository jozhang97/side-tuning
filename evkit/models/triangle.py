import copy
import torch.nn as nn
import torch
from torchsummary import summary

from teas.models.unet import UNet, UNetHeteroscedasticFull, UNetHeteroscedasticIndep, UNetHeteroscedasticPooled


class TriangleModel(nn.Module):

    def __init__(self, network_constructors, n_channels_lists, universal_kwargses=[{}]):
        super().__init__()
        self.chains = nn.ModuleList()
        for network_constructor, n_channels_list, universal_kwargs in zip(network_constructors, n_channels_lists,
                                                                          universal_kwargses):
            print(network_constructor)
            chain = network_constructor(n_channels_list=n_channels_list,
                                        universal_kwargs=universal_kwargs)
            #             chain.append(net)
            self.chains.append(chain)

    def initialize_from_checkpoints(self, checkpoint_paths, logger=None):
        for i, (chain, ckpt_fpath) in enumerate(zip(self.chains, checkpoint_paths)):
            if logger is not None:
                logger.info(f"Loading step {i} from {ckpt_fpath}")
            checkpoint = torch.load(ckpt_fpath)
            sd = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            chain.load_state_dict(sd)
        #         initialize_from_checkpoints(ckpt_fpaths, logger)
        return self

    def forward(self, x):
        chain_outputs = []
        for chain in self.chains:
            outputs = chain(x)
            #             chain_x = x
            #             for net in chain:
            #                 chain_x = net(chain_x)
            #                 outputs.append(chain_x)
            #                 if isinstance(chain_x, tuple):
            #                     chain_x = torch.cat(chain_x, dim=1)
            chain_outputs.append(outputs)
        return chain_outputs
