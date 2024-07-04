import networkx as nx
import random
from classes.PoP import PoP
from classes.SFC import SFC
from classes.Server import Server
from classes.VNF import VNF
from classes.network import Network

pop0 = PoP('PoP0', 0)
pop1 = PoP('PoP1', 1)
pop2 = PoP('PoP2', 2)
pop3 = PoP('PoP3', 3)
pop4 = PoP('PoP4', 4)
pop5 = PoP('PoP5', 5)

vnf_types = ['type1', 'type2', 'type3', 'type4']
pops = [pop1, pop2, pop3, pop4, pop5]

for pop in pops:

    for i in range(10):
        server_load = random.randint(70, 100)
        server = Server('server' + str(i), server_load)
        pop.add_server(server)
network = Network(pops)
network.add_edge(pop0, pop1, **{'allocated_bandwidth': 70})
network.add_edge(pop0, pop2, **{'allocated_bandwidth': 80})
network.add_edge(pop0, pop3, **{'allocated_bandwidth': 78})
network.add_edge(pop0, pop4, **{'allocated_bandwidth': 74})
network.add_edge(pop0, pop5, **{'allocated_bandwidth': 90})

sfc = SFC(pop1, pop4)

sfc.add_vnf('source', 0, random.randint(1, 5))
for i in range(4):
    cpu_demand = random.randint(5, 20)
    bandwidth_demand = random.randint(1, 5)
    vnf_type = vnf_types[random.randint(0, 3)]
    sfc.add_vnf(vnf_type, cpu_demand, bandwidth_demand)

sfc.add_vnf('destination', 0, random.randint(1, 5))

sfc.show_sfc()


