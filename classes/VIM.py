class VIM:

    def __init__(self, id):
        self.id = id
        self.servers = []

    def add_server(self, server):
        self.servers.append(server)

    def north_bound_interface(self):
        pass