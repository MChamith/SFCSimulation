class PoP:

    def __init__(self, id, coordinates):
        self.id = id
        self.servers = []
        self.coorinates = coordinates
        self.vnfs = []

    def add_server(self, server):
        self.servers.append(server)

    def get_available_resources(self):
        total_available = 0
        for server in self.servers:
            total_available += server.get_available_resources()
        return total_available

    def get_allocated_resources(self):
        total_allocated = 0
        for server in self.servers:
            total_allocated += server.allocated_cpu
        return total_allocated

    def add_vnf(self, vnf):
        self.vnfs.append(vnf)
        allocated = False
        for server in self.servers:
            available_cpu = server.get_available_resources()
            if available_cpu > vnf.cpu_demand:
                server.add_vnf(vnf)
                allocated = True

        if allocated:
            print('VNF added in PoP' + str(self.id))
        else:
            print('Not enough CPU to allocate in PoP ' + str(self.id) )
        return allocated

    def get_id(self):
        return self.id

    def get_coordinate(self):
        return self.coorinates




    # def search_pop(self, uuid):
    #     for i in range(self.number):
    #         if (self.nodes[i].get_id() == uuid):
    #             return i
    #         i += 1
    #     return -1
    #
    # def get_pop(self,uuid):
    #     index=self.search_pop(uuid)
    #     if index==-1:
    #         return False
    #     else :
    #         return self.nodes[index]

