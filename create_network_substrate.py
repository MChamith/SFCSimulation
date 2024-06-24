import networkx as nx
import random
from classes.PoP import PoP
from classes.Server import Server
from classes.network import Network

pop1 = PoP('PoP1', (1, 1))
pop2 = PoP('PoP2', (1, 2))
pop3 = PoP('PoP3', (1, 4))
pop4 = PoP('PoP4', (2, 1))
pop5 = PoP('PoP5', (2, 2))
pop6 = PoP('PoP6', (2, 3))

pops = [pop1, pop2, pop3, pop4, pop5, pop6]

for pop in pops:

    for i in range(10):
        server_load = random.randint(70, 100)
        server = Server('server' + str(i), server_load )
        pop.add_server(server)
network = Network(pops)
network.add_edge(pop1, pop4, **{'allocated_resources': 70})
network.add_edge(pop1, pop2, **{'allocated_resources': 80})
network.add_edge(pop4, pop5, **{'allocated_resources': 78})
network.add_edge(pop5, pop6, **{'allocated_resources': 74})
network.add_edge(pop5, pop2, **{'allocated_resources': 90})
network.add_edge(pop6, pop3, **{'allocated_resources': 87})
network.show()
