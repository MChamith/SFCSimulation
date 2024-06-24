class Server:

    def __init__(self, id, cpu):
        self.id = id
        self.allocated_cpu = cpu
        self.vnfs = []

    def get_available_resources(self):
        available_cpu = 100 - self.allocated_cpu
        return available_cpu

    def add_vnf(self, vnf):
        if vnf.cpu_demand < self.get_available_resources():
            self.vnfs.append(vnf)
            self.allocated_cpu += vnf.cpu_demand

