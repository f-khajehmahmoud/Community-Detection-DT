import pandas
from scipy.io import mmread
from scipy.spatial import distance
import math
from numpy import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def Sorting_Payoff_Communities(List_Payoff, CoaSets):
    ar1 = []
    order = np.argsort(List_Payoff)[::-1]
    List_Payoff.sort()
    List_Payoff.reverse()
    for i in range(len(order)):
        ar1 = ar1 + [CoaSets[order[i]]]
    CoaSets = ar1
    CoaSets.reverse()
    return (CoaSets, List_Payoff)


def Calculating_Link(Community, graph):
    N_edge = 0
    noe = 0
    for i in range(len(Community)):
        j = i + 1
        while (j < len(Community)):
            if (Community[i], Community[j]) in graph.edges:
                # N_edge = Mat_weigh[Community[i]][Community[j]] + N_edge
                noe = noe + graph.number_of_edges(Community[i], Community[j])
            j = j + 1
    return noe  # N_edge


def Calculating_Degree(Community, Mat_weigh, graph, E, N_edge):
    N_degree = 0
    # nod = 0
    # for (node, degree) in graph.degree(Community):
    #     nod = nod + degree
    for i in range(len(Community)):
        for j in range(len(Mat_weigh)):
            if ((Community[i], j) in graph.edges and j not in Community):
                N_degree = Mat_weigh[Community[i]][j] + N_degree
    N_degree = N_degree + N_edge
    return N_degree


def Calculating_utility(Community, Mat_weigh, graph, E, a, b):
    N_edge = Calculating_Link(Community, graph)
    N_degree = Calculating_Degree(Community, Mat_weigh, graph, E, N_edge)
    if (N_degree != 0):
        utility = ((2 * N_edge) / N_degree) - a * (math.pow((N_degree / (2 * b * E)), 2))
    else:
        utility = -math.inf
    return utility


def Node_position(i):
    Count_i = 0
    for ii in range(g_nodes_len):
        # print(graph.nodes[ii]['id'])
        if (graph.nodes[ii]['id'] == i):
            break
        else:
            Count_i = Count_i + 1
    return (Count_i)


def do(E, Mat_Weigh):
    print("\n============================================================================================")
    b = 1
    a = 1 / math.sqrt(E)
    List_Payoff = []
    CoaSets = []
    # initializing communitis and payoff
    for i in range(g_nodes_len):
        ar1 = [[i]]
        # print(ar1)
        CoaSets = CoaSets + ar1
        # print(CoaSets)
    for i in range(g_nodes_len):
        P = Calculating_utility(CoaSets[i], Mat_Weigh, graph, E, a, b)
        List_Payoff.append(P)
    CoaSets, List_Payoff = Sorting_Payoff_Communities(List_Payoff, CoaSets)
    print(CoaSets)
    # print(graph.nodes[1])
    G = nx.Graph()
    for n in CoaSets:
        G.add_node(n[0])
    G.add_edges_from(graph.edges)
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True)
    plt.show()
    print(G.nodes(data=True))
    A = nx.adjacency_matrix(G)
    f = open("myfile.txt", "w")
    f.write(str(A))
    f.close()


print("\n*** Welcome to this program ***\n")

graph = nx.karate_club_graph()
# graph = nx.from_scipy_sparse_array(mmread('../soc-dolphins.mtx'))
# graph = nx.read_gml(('../football.gml'))
# graph = nx.convert_node_labels_to_integers(G=graph)

pos = nx.spring_layout(graph)

g_nodes = list(graph.nodes())
g_nodes_len = len(g_nodes)
lst = [(i, i) for i in range(g_nodes_len)]
nx.set_node_attributes(graph, values=dict(lst), name='id')
print(graph.nodes(data=True))

Mat_Weigh = np.zeros((g_nodes_len, g_nodes_len))
E = 0
for i in range(g_nodes_len):
    Count_i = Node_position(i)
    for j in range(g_nodes_len):
        Count_j = Node_position(j)
        if i != j:
            # Mat_Weigh[i][j] = 0
            if (g_nodes[i], g_nodes[j]) in graph.edges:
                # Mat_Weigh[i][j] = 1 / (distance.euclidean(X[Count_i], X[Count_j]))
                Mat_Weigh[i][j] = 1 / (distance.euclidean(pos[Count_i], pos[Count_j]))
            E = Mat_Weigh[i][j] + E
        else:
            Mat_Weigh[i][j] = -math.inf
E = E / 2
do(E, Mat_Weigh)
