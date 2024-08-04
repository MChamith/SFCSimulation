import networkx as nx
import random
from classes.PoP import PoP
from classes.SFC import SFC
from classes.Server import Server
from classes.VNF import VNF
from classes.network import Network
import numpy as np


class PopTopology:
    def __init__(self):
        self.pops = []
        self.network = None

    def initialize_network(self):

        pop0 = PoP('PoP0', coordinates=[2, 2])
        pop1 = PoP('PoP1', coordinates=[1, 1])
        pop2 = PoP('PoP2', coordinates=[1, 3])
        pop3 = PoP('PoP3', coordinates=[3, 3])
        pop4 = PoP('PoP4', coordinates=[3, 1])

        vnf_types = ['type1', 'type2', 'type3', 'type4']
        self.pops = [pop0, pop1, pop2, pop3, pop4]

        for pop in self.pops:

            for i in range(10):
                server_load = random.randint(70, 100)
                server = Server('server' + str(i), server_load)
                pop.add_server(server)
        # Star topology

        self.network = Network(self.pops)
        self.network.add_edge(pop0, pop1, **{'allocated_bandwidth': random.randint(70, 90)})
        self.network.add_edge(pop0, pop2, **{'allocated_bandwidth': random.randint(70, 90)})
        self.network.add_edge(pop0, pop3, **{'allocated_bandwidth': random.randint(70, 90)})
        self.network.add_edge(pop0, pop4, **{'allocated_bandwidth': random.randint(70, 90)})

        return self.network

    def get_all_pop_coordinates(self):
        pop_coordinates = []
        for pop in self.pops:
            pop_coordinates.append(pop.get_coordinate())

        return np.array(pop_coordinates)

    def get_all_pop_cpus(self):
        pop_cpus = []
        for pop in self.network.pop_list:
            pop_cpus.append(pop.get_total_available_resources())
        return np.array(pop_cpus)

    def get_edge_bandwidths(self):
        edge_bandwidths = []
        for edge in self.network.network_graph.edges.data():
            bdw = edge[2]['available_bandwidth']
            edge_bandwidths.append(bdw)

        return np.array(edge_bandwidths)


topology = PopTopology()
network = topology.reset_network()

coords = topology.get_all_pop_coordinates()
print('coords ' + str(coords))

cpus = topology.get_all_pop_cpus()
print('cpus ' + str(cpus))

bdws = topology.get_edge_bandwidths()
print('bandwiodths ' + str(bdws))
# sfc = SFC(pop1, pop4)
#
# sfc.add_vnf('source', 0, random.randint(1, 5))
# for i in range(5):
#     cpu_demand = random.randint(5, 20)
#     bandwidth_demand = random.randint(1, 5)
#     vnf_type = vnf_types[random.randint(0, 3)]
#     sfc.add_vnf(vnf_type, cpu_demand, bandwidth_demand)
#
# sfc.add_vnf('destination', 0, random.randint(1, 5))
#
# # sfc.show_sfc()
#
# sfc_length = sfc.get_sfc_length()
# placements = []
# for i in range(sfc_length):
#     curr_vnf = sfc.get_current_vnf(i)
#     rand_pop_id = random.randint(0, 4)
#     placements.append(rand_pop_id)
#     curr_pop = pops[rand_pop_id]
#     curr_pop.place_vnf(curr_vnf)
#
# # print(placements)
# print('before vlink allocation')
# network.show()
# for i in range(len(placements) - 1):
#     curr_plc = pops[placements[i]]
#     next_plc = pops[placements[i + 1]]
#     bandwidth_req = sfc.get_current_vnf(i).get_bandwidth()
#     network.add_vlink(curr_plc, next_plc, bandwidth_req)
#
# print('after vlink allocation')
#
# network.show()
