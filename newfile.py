import networkx as nx
import networkx.algorithms.community as nx_comm
from sklearn.metrics.cluster import normalized_mutual_info_score

#
# def part_one(main_nodes: int, _b: dict, b: dict, a: list, x: list, ):
#     nodes = (main_nodes // 2) + 1
#     hidden_layer_nodes_range = range(1, nodes)
#     nodes_list = []
#     x_i = []
#     for node in hidden_layer_nodes_range:
#         h = 0
#         for i in hidden_layer_nodes_range:
#             h += a[i] * x[i]
#         h += b[1][node]
#         nodes_list.append(h)
#
#     for i in range(main_nodes):
#         x_i[i] = 0
#         for j in hidden_layer_nodes_range:
#             x_i[i] += _b[j][i] * nodes_list[j] + b[2][i]
#     E = 0
#     for i in range(1, main_nodes + 1):
#         E = 0.5 * (x_i[i] - x[i]) ** 2
#
#     return x_i, E


h = []
a = []
b = []
_b = []
x_output = []

G = nx.karate_club_graph()


def hidden_layer():
    for i in range(0, len(G.nodes) // 2):
        h_i = 0
        for j in range((len(G.nodes) // 2) + 1):
            h_i += a[i] * G.nodes[i]['x']
        h_i += b[1][i]
        h.append(h_i)


def output_layer():
    for i in range(len(G.nodes)):
        x_prim = 0
        for j in range(0, len(G.nodes) // 2):
            x_prim += _b[j][1] * h[j] + b[2][j]
        x_output.append(x_prim)


def sigma(i):
    return x_output[i] - G.nodes[i]['x']


def loss_function():
    error = 0
    for i in range(len(G.nodes)):
        error = 0.5 * sigma(i) ** 2
    return error

