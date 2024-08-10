# #####################################
# ## network类测试文件
# ####################################
# from sfcsim import *
# import matplotlib.pyplot as plt
# import networkx as nx
#
# ##########################节点添加删除测试##########################################
# print(network.__doc__)
# node1 = node(uuid='server1', atts={'cpu': 10, 'memory': 10, 'storage': 10, 'access': False})
# node2 = node(uuid='server2', atts={'cpu': 10, 'memory': 10, 'storage': 10, 'access': False})
# node3 = node(uuid='server3', atts={'cpu': 10, 'memory': 10, 'storage': 10, 'access': False})
# node4 = node(uuid='server4', atts={'cpu': 10, 'memory': 10, 'storage': 10, 'access': False})
# node5 = node(uuid='server5', atts={'cpu': 10, 'memory': 10, 'storage': 10, 'access': False})
# access1 = node(uuid='access1', atts={'access': True})
# access2 = node(uuid='access2', atts={'access': True})
# print('*****************     测试生成网络    ******************')
# nodes1 = [node1, node2, node3, node4, node5]
# network = network(nodes1)
# network.show()
# print('*****************     测试删除节点，删除node1 2 3 4    ******************')
# network.delete_nodes([node2, 'server3', node4, 'server2'])
# network.delete_node(node1)
# network.delete_node(node1)  # 测试重复删除
# print(network.G.nodes)
# network.show()
# print('*****************     测试添加节点，添加node1 2 3 4 5    ******************')
# network.add_node(node5)
# network.add_node(node2)
# network.add_nodes([node1, node3, node4, node5])
# print(network.G.nodes)
# network.show_nodes()
# # print('*****************     测试添加边    ******************')
# # network.add_edge('server1', 'server2', bandwidth=10)
# # network.add_edge(node1, node3, bandwidth=20, delay=10)
# # network.add_edges([['server2', node3, {'bandwidth': 1}], ['server2', 'server4', {'bandwidth': 1}],
# #                    ['server2', 'server6', {'bandwidth': 1}]])
# # network.show_edges()
# # print('***********     测试删除边     ******************')
# # network.delete_edge('server1', node2)
# # network.delete_edge(node1, node4)
# # network.delete_edges([[node1, 'server6'], [node2, 'server4']])
# # network.show_edges()
# # print('***********     测试设置边的属性     ******************')
# # network.add_edge('server1', 'server2', bandwidth=10)
# # network.add_edge('server5', 'server2', bandwidth=10)
# # network.add_edge(node1, node3, bandwidth=20, delay=10)
# # network.add_edge(node1, access1, bandwidth=20, delay=10)
# # network.add_edge(node4, access1, bandwidth=20, delay=10)
# # network.add_edge(node5, access1, bandwidth=20, delay=10)
# # network.add_edge(node4, access2, bandwidth=20, delay=10)
# # network.add_edge(node5, access2, bandwidth=20, delay=10)
# # network.add_edges([['server2', node3, {'bandwidth': 1}], ['server2', 'server4', {'bandwidth': 1}],
# #                    ['server2', 'server6', {'bandwidth': 1}]])
# # network.set_edge_atts('server1', node3, {'bandwidth': 100})
# # network.set_edge_atts('server1', node2, {'bandwidth': 100})
# # network.show_edges()
# # network.set_edges_atts(
# #     {(node1, node2): {'delay': 100}, (node1, 'server3'): {'delay': 100}, (node3, node2): {'delay': 100},
# #      ('server10', node2): {'delay': 100}, })
# # network.show_edges()
# # print('***********     测试画图     ******************')
# # plt.figure(figsize=[25, 15])
# # network.add_vnf('server1', vnf_type(atts={'cpu': 2, 'memory': 5, 'storage': 1}))
# # network.add_vnf('server1', vnf_type(name='type2', atts={'cpu': 2, 'memory': 5, 'storage': 1}))
# # network.add_vnf('server1', vnf_type(name='vnf_type2', atts={'cpu': 1, 'memory': 1, 'storage': 1}))
# # network.add_vnf('server1', vnf_type(atts={'cpu': 1, 'memory': 1, 'storage': 1}))
# # network.add_vnf('server2', vnf_type(atts={'cpu': 1, 'memory': 1, 'storage': 1}))
# # network.show()
# # network.show()
# # network.draw(nx.spring_layout(network.G))

import numpy as np

state = {
    'sfc_length': 5,
    'src_loc': [1, 2],
    'dst_loc': [2, 1],
    'cpu_demand': 34,
    'ingress_bw': 12,
    'egress_bw': 43,
    'order': 4,
    'pop_locations': [[1, 1], [0, 0], [0, 2], [2, 2], [2, 0]],
    'pop_cpu': [3, 4, 5, 6, 6],
    'link_bw': [3, 5, 6, 7],
}

flattened_values = []
for value in state.values():
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                flattened_values.extend(item)
            else:
                flattened_values.append(item)
    else:
        flattened_values.append(value)

# Convert the flattened list to a NumPy array
state_array = np.array(flattened_values)

# Print the resulting array
print(state_array)
