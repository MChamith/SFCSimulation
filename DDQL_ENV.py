import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from network_substrate import PopTopology
import numpy as np


class DdqlEnv(Env):

    def __init__(self, pop_n, edge_n, coord_range, sfc_dataset):
        super(DdqlEnv, self).__init__()

        self.pop_n = pop_n  # Number of Pops or number of actions in action space
        self.edge_n = edge_n
        self.coord_range = coord_range
        self.sfc_dataset = sfc_dataset
        self.curr_sfc_no = 0
        self.pop_topology = PopTopology()
        self.physical_network = self.pop_topology.initialize_network()
        self.curr_sfc = None
        self.vnf_order = 0
        self.placements = []

        self.observation_space = spaces.Dict({
            'sfc_length': spaces.Discrete(5),  # set to max sfc length with src dst included
            'src_loc': spaces.Box(low=0, high=self.coord_range, shape=(2,), dtype=np.int8),
            'dst_loc': spaces.Box(low=0, high=self.coord_range, shape=(2,), dtype=np.int8),
            # VNF state: dR(i), dR(i-, i), dR(i, i+), i.order
            'cpu_demand': spaces.Box(low=0, high=100, shape=(1,)),
            'ingress_bw': spaces.Box(low=0, high=100, shape=(1,)),
            'egress_bw': spaces.Box(low=0, high=100, shape=(1,)),
            'order': spaces.Discrete(5),  # set to max sfc length with src dst included
            # Substrate state: u.loc, dS,t(u), dS,t(u, v)
            'pop_locations': spaces.Box(low=0, high=self.coord_range, shape=(self.pop_n, 2), dtype=np.int8),
            'pop_cpu': spaces.Box(low=0, high=100, shape=(self.pop_n,)),
            'link_bw': spaces.Box(low=0, high=100, shape=(self.edge_n,)),
        })

        self.action_space = spaces.Discrete(self.pop_n)

    def step(self, action):

        done = False
        PoP = self.physical_network.get_pop_by_action(action)
        vnf = self.curr_sfc.get_vnf(self.vnf_order)
        n_success = PoP.place_vnf(vnf)
        self.placements.insert(self.vnf_order, PoP)
        # TODO check below maps correctly
        # print('vnf order ' + str(self.vnf_order))
        # print('terminal ' + str(self.sfc_length - 2))
        is_vnf_terminal = self.vnf_order == self.sfc_length - 2
        if n_success and is_vnf_terminal:
            l_success, act_path = self.pop_topology.place_vlinks(self.curr_sfc, self.placements)
            if l_success:
                opt_path = self.pop_topology.calculate_opt_path(self.placements[0], self.placements[-1])
                # TODO check the path length in edge case when vnf in single pop. src = dst
                reward = 10 * ((opt_path + 1) / (act_path + 1))
                print('success act path ' + str(act_path) + ' opt path ' + str(opt_path))

            else:
                reward = -10
            done = True

        elif n_success:
            reward = 0.1
            self.vnf_order += 1
            self.state = self.get_curr_state()
        else:
            reward = -10
            done = True

        return self.state, reward, done, {}

    def reset(self):

        self.physical_network = self.pop_topology.initialize_network()
        self.curr_sfc = self.sfc_dataset[self.curr_sfc_no]
        self.vnf_order = 1
        self.sfc_length = self.curr_sfc.get_sfc_length()
        self.placements = []

        sourcPoP = self.physical_network.get_pop_by_coordinates(self.curr_sfc.get_source())
        source_vnf = self.curr_sfc.get_vnf(0)
        sourcPoP.place_vnf(source_vnf)
        self.placements.append(sourcPoP)

        destPoP = self.physical_network.get_pop_by_coordinates(self.curr_sfc.get_destination())
        dest_vnf = self.curr_sfc.get_vnf(self.sfc_length - 1)
        destPoP.place_vnf(dest_vnf)
        self.placements.append(destPoP)

        self.state = self.get_curr_state()

        self.curr_sfc_no += 1

        return self.state

    def get_curr_state(self):

        curr_vnf = self.curr_sfc.get_vnf(self.vnf_order)
        prev_vnf = self.curr_sfc.get_vnf(self.vnf_order - 1)

        self.state = {
            'sfc_length': self.sfc_length,
            'src_loc': self.curr_sfc.get_source(),
            'dst_loc': self.curr_sfc.get_destination(),
            'cpu_demand': curr_vnf.get_cpu_demand(),
            'ingress_bw': prev_vnf.get_bandwidth(),
            'egress_bw': curr_vnf.get_bandwidth(),
            'order': self.vnf_order,
            'pop_locations': self.pop_topology.get_all_pop_coordinates(),
            'pop_cpu': self.pop_topology.get_all_pop_cpus(),
            'link_bw': self.pop_topology.get_edge_bandwidths(),
        }

        return self.state
