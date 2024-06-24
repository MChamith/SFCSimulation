import networkx as nx
from classes.PoP import PoP
from classes.network import Network
pop1 = PoP('PoP1', (1, 1))
pop2 = PoP('PoP2', (1, 2))
pop3 = PoP('PoP3', (1, 1))
pop4 = PoP('PoP4', (1, 2))
pop5 = PoP('PoP5', (1, 1))
pop6 = PoP('PoP6', (1, 2))

nodes1 = [pop1, pop2, pop3, pop4, pop5, pop6]
network = network(nodes1)
network.show()