from classes.SFC import SFC
import random
from DDQL_ENV import DdqlEnv

coordinate_list = [[1, 1], [0, 0], [0, 2], [2, 2], [2, 0]]
sfc1 = SFC(coordinate_list[2], coordinate_list[4])
sfc2 = SFC(coordinate_list[0], coordinate_list[3])
vnf_types = ['type1', 'type2', 'type3', 'type4']

for i in range(5):
    if i == 0:
        cpu_demand = 0
        vnf_type = 'source'
    else:
        cpu_demand = random.randint(5, 20)
        vnf_type = vnf_types[random.randint(0, 3)]
    bandwidth_demand = random.randint(1, 5)

    sfc1.add_vnf(vnf_type, cpu_demand, bandwidth_demand)

for i in range(5):
    if i == 0:
        cpu_demand = 0
        vnf_type = 'source'
    else:
        cpu_demand = random.randint(5, 20)
        vnf_type = vnf_types[random.randint(0, 3)]
    bandwidth_demand = random.randint(1, 5)

    sfc2.add_vnf(vnf_type, cpu_demand, bandwidth_demand)

sfc_dataset = [sfc1, sfc2]

env = DdqlEnv(5, 4, 2, sfc_dataset )

for sfc in sfc_dataset:
    done = False
    state = env.reset()
    score = 0

    while not done:
        action = env.action_space.sample()
        new_state, reward, done, _ = env.step(action)
        score += reward
