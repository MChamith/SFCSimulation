from classes.SFC import SFC
import random
from DDQL_ENV import DdqlEnv
import gymnasium as gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR


class QNetwork(nn.Module):
    def __init__(self, V_s, E_s):
        super(QNetwork, self).__init__()
        input_dim = 3*V_s + E_s + 9
        self.fc_1 = nn.Linear(input_dim, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, V_s)

    def forward(self, inp):
        x1 = F.relu(self.fc_1(inp))
        x1 = F.relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1
