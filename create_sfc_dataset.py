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


def create_dataset(sfc_length):
    print('creating sfc dataset')
    coordinate_list = [[0, 0], [0, 2], [1, 1], [2, 2], [2, 0]]
    vnf_types = ['type1', 'type2', 'type3', 'type4']
    sfc_dataset = []
    for i in range(sfc_length):
        sfc = SFC(random.choice(coordinate_list), random.choice(coordinate_list))

        num_vnfs = random.randint(4, 7)
        for i in range(num_vnfs):
            if i == 0:
                cpu_demand = 0
                vnf_type = 'source'
                bandwidth_demand = random.randint(1, 5)
            elif i == num_vnfs -1:
                cpu_demand = 0
                vnf_type = 'destination'
                bandwidth_demand = 0
            else:
                cpu_demand = random.randint(5, 20)
                vnf_type = vnf_types[random.randint(0, 3)]
                bandwidth_demand = random.randint(1, 5)
            sfc.add_vnf(vnf_type, cpu_demand, bandwidth_demand)
            sfc_dataset.append(sfc)

    return sfc_dataset

# dts = create_dataset()
# print(dts)