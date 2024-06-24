import networkx as nx
from classes.PoP import PoP


class Network:

    def __init__(self, pops=None, edges=None):
        if pops is None:
            pops = []
        if edges is None:
            edges = []
        self.pop_list = pops
        self.edges = edges
        self.network_graph = nx.Graph()

    def generate(self):
        for pop in self.pop_list:  # 如果nodes的节点不再G中，则加入节点
            if pop not in self.network_graph:
                self.network_graph.add_node(pop)
        n = []
        for pop in self.network_graph.nodes:  # 如果G中的节点不再nodes中，则删除G中节点，一切以nodes中为准
            if pop not in self.pop_list:
                n.append(pop)
        self.network_graph.remove_nodes_from(n)

    def add_pop(self, pop):
        self.pop_list.append(pop)
        self.generate()

    def remove_pop(self, pop):
        self.pop_list.remove(pop)
        self.generate()

    def add_edge(self, pop1, pop2, **link):
        link['available_resources'] = 100 - link['allocated_resources']
        self.network_graph.add_edge(pop1, pop2, **link)

    def show_nodes(self):
        print('*****     there are', len(self.network_graph.nodes), 'node in network     *****')
        print('    number  PoP       coordinates       available Resources ')
        i = 1
        for pop in self.pop_list:
            print('    %-6d  %-10s  %-10s  %-s' % (i, pop.get_id(), pop.get_coordinate(), pop.get_available_resources()))
            i+=1
        # print(self.pop_list)

    def show_edges(self):
        i = 1
        print('*****     there are', len(self.network_graph.edges), 'edge in network     *****')
        print('    number  node1       node2       atts')
        for edge in self.network_graph.edges.data():
            print('    %-6d  %-10s  %-10s  %-s' % (i, edge[0].get_id(), edge[1].get_id(), edge[2]))
            i += 1

    def show(self):
        print('**********************     print nodes and edges in network     *********************')
        self.show_nodes()
        self.show_edges()
