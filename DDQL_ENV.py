import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from network_substrate import PopTopology


class DdqlEnv(Env):

    def __init__(self, sfc_length, pop_n, edge_n, coord_range, sfc_dataset):
        super(DdqlEnv, self).__init__()

        self.sfc_length = sfc_length
        self.pop_n = pop_n  # Number of Pops or number of actions in action space
        self.edge_n = edge_n
        self.coord_range = coord_range
        self.sfc_dataset = sfc_dataset
        self.curr_sfc_no = 0
        self.pop_topology = PopTopology()
        self.physical_network = self.pop_topology.initialize_network()
        self.curr_sfc = None
        self.vnf_order = 0

        self.observation_space = spaces.Dict({
            'sfc_length': spaces.Discrete(self.sfc_length),
            'src_loc': spaces.Box(low=0, high=self.coord_range, shape=(2,)),
            'dst_loc': spaces.Box(low=0, high=self.coord_range, shape=(2,)),
            # VNF state: dR(i), dR(i-, i), dR(i, i+), i.order
            'cpu_demand': spaces.Box(low=0, high=100, shape=(1,)),
            'ingress_bw': spaces.Box(low=0, high=100, shape=(1,)),
            'egress_bw': spaces.Box(low=0, high=100, shape=(1,)),
            'order': spaces.Discrete(self.sfc_length),
            # Substrate state: u.loc, dS,t(u), dS,t(u, v)
            'pop_locations': spaces.Box(low=0, high=self.coord_range, shape=(self.pop_n, 2)),
            'pop_cpu': spaces.Box(low=0, high=100, shape=(self.pop_n,)),
            'link_bw': spaces.Box(low=0, high=100, shape=(self.edge_n,)),
        })

        self.action_space = spaces.Discrete(self.pop_n)

    def step(self, action):
        # done = False
        # PoP = self.physical_network.get_pop_by_action(action)
        # vnf = self.curr_sfc.get_vnf(self.vnf_order)
        # n_success = PoP.place_vnf(vnf)
        # #TODO check below maps correctly
        # is_vnf_terminal = self.vnf_order == self.sfc_length - 2
        # if n_success and is_vnf_terminal:
        #     pass
        #
        # elif n_success:
        #     reward = 0.1
        #     self.vnf_order += 1
        # else:
        #     reward = -10
        #     done = True
        #
        # return
        pass

    def reset(self):
        self.physical_network = self.pop_topology.initialize_network()
        self.curr_sfc = self.sfc_dataset[self.curr_sfc_no]
        self.vnf_order = 0
        self.state = {
            'sfc_length': self.sfc_length,
            'src_loc': self.curr_sfc.get_source(),
            'dst_loc': self.curr_sfc.get_destination(),
            'cpu_demand': 0,
            'ingress_bw': 0,
            'egress_bw': self.curr_sfc.get_vnf(0).get_bandwidth(),
            'order': 0,
            'pop_locations': self.pop_topology.get_all_pop_coordinates(),
            'pop_cpu': self.pop_topology.get_all_pop_cpus(),
            'link_bw': self.pop_topology.get_edge_bandwidths(),
        }

        self.curr_sfc_no += 1

        return self.state
