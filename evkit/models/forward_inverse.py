from gym import spaces
import multiprocessing.dummy as mp
import multiprocessing
import numpy as np
import os
import torch
import torch
import torch.nn as nn
from   torch.nn import Parameter, ModuleList
import torch.nn.functional as F

from evkit.rl.utils import init, init_normc_
from evkit.utils.misc import is_cuda
from evkit.preprocess import transforms

import pickle as pkl

init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              nn.init.calculate_gain('relu'))

################################
# Inverse Models
#   Predict  s_{t+1} | s_t, a_t
################################
class ForwardModel(nn.Module):
    
    def __init__(self, state_shape, action_shape, hidden_size):
        super().__init__()
        self.fc1 = init_(nn.Linear(state_shape + action_shape[1], hidden_size))
        self.fc2 = init_(nn.Linear(hidden_size, state_shape))
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

################################
# Inverse Models
#   Predict a_t | s_t, s_{t+1}
################################
class InverseModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = init_(nn.Linear(input_size * 2, hidden_size))
        # Note to stoip gradient
        self.fc2 = init_(nn.Linear(hidden_size, output_size))

    def forward(self, phi_t, phi_t_plus_1):
        x = torch.cat([phi_t, phi_t_plus_1], 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
#         ainvprobs = nn.softmax(logits, dim=-1)