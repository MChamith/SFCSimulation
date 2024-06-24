class VNF:

    def __init__(self, vnf_type, cpu_demand):
        self.vnf_type = vnf_type
        self.cpu_demand = cpu_demand

    def get_vnf_type(self):
        return self.vnf_type

