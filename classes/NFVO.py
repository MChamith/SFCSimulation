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




env = DdqlEnv(5, 4, 2, sfc_dataset )

# for sfc in sfc_dataset:
#     done = False
#     state = env.reset()
#     score = 0
#
#     while not done:
#         action = env.action_space.sample()
#         new_state, reward, done, _ = env.step(action)
#         score += reward
