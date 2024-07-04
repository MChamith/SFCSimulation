class VNF:

    def __init__(self, vnf_type, cpu_demand, bandwidth, order):
        self.vnf_type = vnf_type
        self.cpu_demand = cpu_demand
        self.bandwidth = bandwidth
        self.order = order

    def get_vnf_type(self):
        return self.vnf_type

    def get_cpu_demand(self):
        return self.cpu_demand

    def get_bandwidth(self):
        return self.bandwidth

