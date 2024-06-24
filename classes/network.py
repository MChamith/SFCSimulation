import networkx as nx

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
            if node not in self.pop_list:
                n.append(node)
        self.network_graph.remove_nodes_from(n)

    def add_pop(self, pop):
        self.pop_list.append(pop)
        self.generate()

    def remove_pop(self, pop):
        self.pop_list.remove(pop)
        self.generate()

    def add_edge(self, pop1, pop2):
