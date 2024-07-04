from classes.VNF import VNF


class SFC:

    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.vnfs = []
        self.vnf_order = 0

    def add_vnf(self, vnf_type, cpu_demand, bandwidth_demand):
        vnf = VNF(vnf_type, cpu_demand, bandwidth_demand, self.vnf_order)
        self.vnfs.append(vnf)
        self.vnf_order += 1

    def get_source(self):
        return self.source

    def get_destination(self):
        return self.destination

    def get_sfc_length(self):
        return len(self.vnfs)

    def show_sfc(self):
        for i in range(len(self.vnfs)):
            vnf = self.vnfs[i]
            print('VNF type ' + str(vnf.get_vnf_type()) + ', cpu demand ' + str(vnf.get_cpu_demand()) +
                  ', bandwidth demand ' + str(vnf.get_bandwidth()) + ' order ' + str(i))
