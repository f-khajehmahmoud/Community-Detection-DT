import sys
import math
import random

from scipy.io import mmread
import seaborn as sns
from numpy import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

import pandas
from scipy.spatial import distance
import math
import pagerank

from numpy import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class ConsoleTextColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


import pandas
from scipy.spatial import distance
import math
from numpy import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

print('Please select dataset to continue:')
sets_name_list = [
    # f'{ConsoleTextColors.OKBLUE}1{ConsoleTextColors.ENDC}-network19',
    f'{ConsoleTextColors.OKGREEN}1{ConsoleTextColors.ENDC}-karate',
    f'{ConsoleTextColors.OKCYAN}2{ConsoleTextColors.ENDC}-dolphins',
    f'{ConsoleTextColors.WARNING}3{ConsoleTextColors.ENDC}-football',
]
sets_simple_name_list = [
    # 'network19',
    'karate   ',
    'dolphins ',
    'football ',
]
for name in sets_name_list:
    print(name)

print(name)

dataset = int(input('Enter dataset index:')) - 1
print('selected dataset= ', sets_name_list[dataset])
if dataset == 0:
    # G = nx.from_scipy_sparse_array(mmread('soc-karate.mtx'))
    G = nx.karate_club_graph()
elif dataset == 1:
    G = nx.from_scipy_sparse_array(mmread('soc-dolphins.mtx'))
else:
    G = nx.read_gml(('football.gml'))
    G = nx.convert_node_labels_to_integers(G)

nx.draw_networkx(G)
plt.show()


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
    noe = 0
    for i in range(len(Community)):
        j = i + 1
        while (j < len(Community)):
            if (Community[i], Community[j]) in graph.edges:
                noe = noe + graph.number_of_edges(Community[i], Community[j])
            j = j + 1
    return noe  # N_edge


def Calculating_Degree(Community, Mat_weigh, graph, E, N_edge):
    N_degree = 0
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
        if (G.nodes[ii]['id'] == i):
            break
        else:
            Count_i = Count_i + 1
    return (Count_i)


def do(E, Mat_Weigh):
    b = 1
    a = 1 / math.sqrt(E)
    List_Payoff = []
    CoaSets = []
    for i in range(g_nodes_len):
        ar1 = [[i]]
        CoaSets = CoaSets + ar1
    for i in range(g_nodes_len):
        P = Calculating_utility(CoaSets[i], Mat_Weigh, G, E, a, b)
        List_Payoff.append(P)
    CoaSets, List_Payoff = Sorting_Payoff_Communities(List_Payoff, CoaSets)
    G = nx.Graph()
    for n in CoaSets:
        G.add_node(n[0])
    G.add_edges_from(G.edges)
    nx.draw(G, pos=nx.circular_layout(G), with_labels=True)
    plt.show()
    A = nx.adjacency_matrix(G)
    f = open("myfile.txt", "w")
    f.write(str(A))
    f.close()


pos = nx.spring_layout(G)
g_nodes = list(G.nodes())
g_nodes_len = len(g_nodes)
lst = [(i, i) for i in range(g_nodes_len)]
nx.set_node_attributes(G, values=dict(lst), name='id')

Mat_Weigh = np.zeros((g_nodes_len, g_nodes_len))
E = 0
for i in range(g_nodes_len):
    Count_i = Node_position(i)
    for j in range(g_nodes_len):
        Count_j = Node_position(j)
        if i != j:
            if (g_nodes[i], g_nodes[j]) in G.edges:
                Mat_Weigh[i][j] = 1 / (distance.euclidean(pos[Count_i], pos[Count_j]))
            E = Mat_Weigh[i][j] + E
        else:
            Mat_Weigh[i][j] = -math.inf
E = E / 2
do(E, Mat_Weigh)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

h = []
a = []
b = []
_b = []
x_output = []



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


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


def sigmoid(x):
    return math.tanh(x)


def dsigmoid(y):
    return 1.0 - y ** 2


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            # self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change
                # print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
import networkx.algorithms.community as nx_comm

print(nx_comm.label_propagation_communities(G))
print(f'{ConsoleTextColors.OKBLUE}modularity{ConsoleTextColors.ENDC}',
      nx_comm.modularity(G, nx_comm.label_propagation_communities(G)))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.metrics.cluster import normalized_mutual_info_score

for i in range(1, 15):
    kmeans_i = KMeans(n_clusters=i, random_state=0).fit(data)
    print(f'{ConsoleTextColors.OKGREEN}NMI : {ConsoleTextColors.FAIL}{i}{ConsoleTextColors.ENDC}->',
          normalized_mutual_info_score(kmeansi.labels_, kmeans_i.labels_))
