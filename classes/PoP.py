from classes.MARLAgent import Agent


class PoP:

    def __init__(self, id, coordinates):
        self.id = id
        self.servers = []
        self.coorinates = coordinates
        self.vnfs = []
        self.vnf_id = []
        self.agent = None

    def add_server(self, server):
        self.servers.append(server)

    def get_total_available_resources(self):
        total_available = 0
        server_count = 0
        for server in self.servers:
            total_available += server.get_available_resources()
            server_count += 1
        return total_available / server_count

    def get_total_allocated_resources(self):
        total_allocated = 0
        server_count = 0
        for server in self.servers:
            total_allocated += server.allocated_cpu
            server_count += 1
        return total_allocated

    def place_vnf(self, vnf, vnf_id):
        # print('available before allocation ' + str(self.get_total_available_resources()))
        self.vnfs.append(vnf)
        allocated = False
        for server in self.servers:
            available_cpu = server.get_available_resources()
            if available_cpu > vnf.cpu_demand:
                server.add_vnf(vnf)
                allocated = True
                self.vnf_id.append(vnf_id)
                break

        if not allocated:
            print('Not enough CPU to allocate in PoP ' + str(self.id))

        # print('after allocation ' + str(self.get_total_available_resources()))
        return allocated

    def get_id(self):
        return self.id

    def get_coordinate(self):
        return self.coorinates

    def get_latest_vnf(self):
        if len(self.vnf_id) > 0:
            return self.vnf_id[-1]
        else:
            return -1

    def initialize_agent(self, topology, n_actions, id, mem_size):
        self.agent = Agent(topology, n_actions, id, mem_size)
        return self.agent

    def get_local_cpus(self):
        local_cpu = []
        for server in self.servers:
            local_cpu.append(server.get_available_resources())

        return local_cpu

    def get_number_of_servers(self):
        return len(self.servers)



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
