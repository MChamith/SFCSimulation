class PoP:

    def __init__(self, id, coordinates):
        self.id = id
        self.servers = []
        self.coorinates = coordinates

    def assign_server(self, server):
        self.servers.append(server)

    def get_available_resources(self):
        pass

    def get_allocated_resources(self):
        pass
