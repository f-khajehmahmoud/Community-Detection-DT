import networkx as nx
from scipy.spatial import distance
import math
import numpy as np


def Sorting_Payoff_Communities(List_Payoff, CoaSets):
    ar1 = []
    order = np.argsort(List_Payoff)[::-1]
    List_Payoff.sort()
    List_Payoff.reverse()
    for i in range(len(order)):
        ar1 = ar1 + [CoaSets[order[i]]]
    CoaSets = ar1
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
    for ii in range(len(graph.nodes())):
        if (Node_pos[ii] == i):
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
    for i in range(len(graph.nodes)):
        ar1 = [[i]]
        # print(ar1)
        CoaSets = CoaSets + ar1
        # print(CoaSets)
    for i in range(len(graph.nodes)):
        P = Calculating_utility(CoaSets[i], Mat_Weigh, graph, E, a, b)
        List_Payoff.append(P)
    CoaSets, List_Payoff = Sorting_Payoff_Communities(List_Payoff, CoaSets)
    print(List_Payoff)
    print(CoaSets)
    # ar_b = []
    # while (1):
    #     i = 0
    #     while (i < len(CoaSets)):
    #         Community = []
    #         PayOff_Total = -math.inf
    #         j = i + 1
    #         while (j < len(CoaSets)):
    #             k = 0
    #             T = 0
    #             while (k < len(CoaSets[i])):
    #                 kk = 0
    #                 while (kk < len(CoaSets[j])):
    #                     if ((CoaSets[i][k], CoaSets[j][kk]) in graph.edges or (CoaSets[j][kk], CoaSets[i][k]) in graph.edges):
    #                         C = CoaSets[i] + CoaSets[j]
    #                         P = Calculating_utility(C, Mat_Weigh, graph, E, a, b)
    #                         if (P > PayOff_Total):
    #                             PayOff1 = List_Payoff[i]
    #                             PayOff2 = List_Payoff[j]
    #                             Community = C
    #                             PayOff_Total = P
    #                             J = j
    #                         T = 1
    #                         break
    #                     else:
    #                         kk = kk + 1
    #                 if (T == 1):
    #                     break
    #                 k = k + 1
    #             j = j + 1
    #         N_edge = Calculating_Link(Community, graph)
    #         if (PayOff_Total >= PayOff1 and PayOff_Total >= PayOff2 and N_edge <= math.sqrt(2 * b * E)):
    #             CoaSets[i] = Community
    #             List_Payoff[i] = PayOff_Total
    #             del List_Payoff[J]
    #             del CoaSets[J]
    #         else:
    #             i = i + 1
    #     CoaSets, List_Payoff = Sorting_Payoff_Communities(List_Payoff, CoaSets)
    #     if (CoaSets == ar_b):
    #         print("\n################# Result #################")
    #         print(CoaSets)
    #         print("##########################################\n")
    #         break
    #     else:
    #         ar_b = CoaSets


print("\n*** Welcome to this program ***\n")
graph = nx.karate_club_graph()

X = np.loadtxt("embeddings.emb", skiprows=1)
Node_pos = np.array([int(x[0]) for x in X])
X = np.array([x[1:] for x in X])
# Mat_Weigh = np.zeros((len(graph.nodes()), len(graph.nodes())))
# E = 0
# for i in range(len(X)):
#     Count_i = Node_position(i)
#     for j in range(len(X)):
#         Count_j = Node_position(j)
#         if i != j:
#             Mat_Weigh[i][j] = 1 / (distance.euclidean(X[Count_i], X[Count_j]))
#             E = Mat_Weigh[i][j] + E
#         else:
#             Mat_Weigh[i][j] = -math.inf
#
# # print('Mat_Weigh=', Mat_Weigh)
# # print('E', E)
# E = E / 2
# print(f"\n==> Calculated E From New Program: {E}")
#
# do(E, Mat_Weigh)


g_nodes = list(graph.nodes())
g_nodes_len = len(g_nodes)
Mat_Weigh = np.zeros((g_nodes_len, g_nodes_len))
print(f"\n==> First Mat Weigh: \n{np.matrix(Mat_Weigh)}")
E = 0
for i in range(g_nodes_len):
    Count_i = Node_position(i)
    for j in range(g_nodes_len):
        Count_j = Node_position(j)
        if i != j:
            Mat_Weigh[i][j] = 0
            if (g_nodes[i], g_nodes[j]) in graph.edges:
                Mat_Weigh[i][j] = 1 / (distance.euclidean(X[Count_i], X[Count_j]))
            E = Mat_Weigh[i][j] + E
        else:
            Mat_Weigh[i][j] = -math.inf
print(f"\n==> Second Mat Weigh: \n{np.matrix(Mat_Weigh)}")
E = E / 2
print(f"\n==> Calculated E From Old Program: {E}")
do(E, Mat_Weigh)
