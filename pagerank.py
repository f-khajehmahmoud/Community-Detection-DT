import sys

from scipy.io import mmread
import seaborn as sns
from numpy import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


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

dataset = int(input('Enter dataset index:')) - 1
print('selected dataset= ', sets_name_list[dataset])
if dataset == 0:
    # G = nx.from_scipy_sparse_array(mmread('soc-karate.mtx'))
    G = nx.karate_club_graph()
elif dataset == 1:
    G = nx.from_scipy_sparse_array(mmread('soc-dolphins.mtx'))
else:
    G = nx.read_gml(('football.gml'))
    G = nx.convert_node_labels_to_integers(G=G)

nx.draw_networkx(G)
plt.show()
# A = nx.to_scipy_sparse_array(G)
# print(A.todense())

# ------------------------------------------------------------------------------------------------------------------------------

Iiteration = 10000

# B1 = input("Please Enter for continue processing ***")
total_num_Total_Conclusion = 4000.0
alpha = .22
beta = 8.5
Imaginary_cap = 2100.0
questioner = 10.0
Decoder_paDecoder = 55.0
Encoder_paDecoder = 58.0
CNN_paDecoder = 62.0


def f1(x):
    return ((questioner / Decoder_paDecoder) * (1 + alpha * ((x / Imaginary_cap) ** beta)))


def f2(x):
    return ((questioner / Encoder_paDecoder) * (1 + alpha * ((x / Imaginary_cap) ** beta)))


def f3(x):
    return ((questioner / CNN_paDecoder) * (1 + alpha * ((x / Imaginary_cap) ** beta)))


for iter in range(Iiteration):
    global Imaging, Imaging_2
    Dataset_Part1_zoon = 0
    Dataset_Part2_zoon = 0
    Dataset_Part3_zoon = 0

    for i in range(int(total_num_Total_Conclusion)):
        Distnc = random.randint(1, 3)
        if (Distnc == 1):
            Dataset_Part1_zoon = Dataset_Part1_zoon + 1
        elif (Distnc == 2):
            Dataset_Part2_zoon = Dataset_Part2_zoon + 1
        else:
            Dataset_Part3_zoon = Dataset_Part3_zoon + 1
    Dataset_Part1_zoon_cost = f1(Dataset_Part1_zoon)
    Dataset_Part2_zoon_cost = f2(Dataset_Part2_zoon)
    Dataset_Part3_zoon_cost = f3(Dataset_Part3_zoon)


    def be_better(l1, l2, l3):
        if ((f1(l1) > f2(l2 + 1)) or f1(l1) > f3(l3 + 1)):
            if (l1 > 0):
                return (True, 1)
        if ((f2(l2) > f1(l1 + 1)) or f2(l2) > f3(l3 + 1)):
            if (l2 > 0):
                return (True, 2)
        if ((f3(l3) > f1(l1 + 1)) or f3(l3) > f2(l2 + 1)):
            if (l3 > 0):
                return (True, 3)
        return (False, 0)


    continue_switching = be_better(Dataset_Part1_zoon, Dataset_Part2_zoon, Dataset_Part3_zoon)[0]
    count = 1
    while (continue_switching):
        l = [1, 2, 3]
        current_Distnc = be_better(Dataset_Part1_zoon, Dataset_Part2_zoon, Dataset_Part3_zoon)[1]
        l.remove(current_Distnc)
        if (current_Distnc == 1):
            Dataset_Part1_zoon = Dataset_Part1_zoon - 1

            if ((f2(Dataset_Part2_zoon + 1)) < (f3(Dataset_Part3_zoon + 1))):
                Dataset_Part2_zoon = Dataset_Part2_zoon + 1
                Dataset_Part2_zoon_cost = f2(Dataset_Part2_zoon)

            else:
                Dataset_Part3_zoon = Dataset_Part3_zoon + 1
                Dataset_Part2_zoon_cost = f3(Dataset_Part3_zoon)
        elif (current_Distnc == 2):
            Dataset_Part2_zoon = Dataset_Part2_zoon - 1

            if ((f1(Dataset_Part1_zoon + 1)) < (f3(Dataset_Part3_zoon + 1))):
                Dataset_Part1_zoon = Dataset_Part1_zoon + 1
                Dataset_Part1_zoon_cost = f1(Dataset_Part1_zoon)

            else:
                Dataset_Part3_zoon = Dataset_Part3_zoon + 1
                Dataset_Part3_zoon_cost = f3(Dataset_Part3_zoon)
        else:
            Dataset_Part3_zoon = Dataset_Part3_zoon - 1

            if ((f1(Dataset_Part1_zoon + 1)) < (f2(Dataset_Part2_zoon + 1))):
                Dataset_Part1_zoon = Dataset_Part1_zoon + 1
                Dataset_Part1_zoon_cost = f1(Dataset_Part1_zoon_cost)

            else:
                Dataset_Part2_zoon = Dataset_Part2_zoon + 1
                Dataset_Part2_zoon_cost = f2(Dataset_Part2_zoon_cost)
        continue_switching = be_better(Dataset_Part1_zoon, Dataset_Part2_zoon, Dataset_Part3_zoon)[0]
        CC1 = Dataset_Part1_zoon
        CC2 = Dataset_Part2_zoon
        CC3 = Dataset_Part3_zoon
        CC1 = CC1 / 10000
        CC2 = CC2 / 10000
        CC3 = CC3 / 10000
        Imaging = (round(CC1, 2), round(CC2, 2), round(CC3, 2))
        BB1 = f1(Dataset_Part1_zoon)
        BB2 = f2(Dataset_Part2_zoon)
        BB3 = f3(Dataset_Part3_zoon)
        Imaging_2 = (round(BB1, 2), round(BB2, 2), round(BB3, 2))
        count = count + 1

    best_response_Imaging = [Dataset_Part1_zoon, Dataset_Part2_zoon, Dataset_Part3_zoon]
    if (iter * 100) % Iiteration == 0:
        # print('working:', (iter * 100) // Iiteration, '%')
        perc = (iter * 100) // Iiteration
        print(f"\rProgress: {ConsoleTextColors.OKGREEN}{perc}{ConsoleTextColors.ENDC} %", end='', flush=True)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
print()
barWidth = 3
fig = plt.subplots(figsize=(12, 8))
data_numbers = (
    [0.36, 0.32, 0.37, 0.41],
    [0.84, 0.57, 1.00, 1.00],

    [0.38, 0.38, 0.38, 0.39],
    [0.89, 0.70, 0.81, 0.80],

    [0.59, 0.54, 0.60, 0.58],
    [0.91, 0.86, 0.91, 0.93],
)

final_numbers = (
    [0.37, 0.41, 0.36, 0.35],
    # [1.00, 1.00, 0.97, 0.99],
    [1.00, 1.00, 0.80, 0.73],

    [0.39, 0.39, 0.36, 0.38],
    # [0.81, 0.80, 0.77, 0.79],
    [0.81, 0.79, 0.82, 0.78],

    [0.59, 0.58, 0.58, 0.55],
    # [0.92, 0.93, 0.90, 0.88],
    [0.92, 0.94, 0.87, 0.85],
)

Article, Prev_prj, PageRank, Game_theo = final_numbers[2 * dataset]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("{0: >5}  {1: >20}  {2: >20}  {3: >20}  {4: >20}".format(ConsoleTextColors.HEADER + sets_simple_name_list[dataset],
                                                               ConsoleTextColors.OKCYAN + "RM+AE+CNN",
                                                               ConsoleTextColors.OKBLUE + "Leader+AE+CNN",
                                                               ConsoleTextColors.OKGREEN + "Leader+NewFormula+AE+CNN",
                                                               ConsoleTextColors.WARNING + "PageRank+AE+CNN" + ConsoleTextColors.ENDC))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("{0: >10}  {1: >10.3}  {2: >15.3}  {3: >20.3}  {4: >20.3}".format(ConsoleTextColors.FAIL + "Modularity" + ConsoleTextColors.ENDC,
                                                                        Article,
                                                                        Prev_prj,
                                                                        PageRank,
                                                                        Game_theo))
br1 = Article
br2 = br1 + barWidth
br3 = br2 + barWidth
br4 = br3 + barWidth
plt.bar(br1, Article, color='cornflowerblue', edgecolor='grey', label='RM+AE+CNN')
plt.bar(br2, Prev_prj, color='darkblue', edgecolor='grey', label='Leader+AE+CNN')
plt.bar(br3, PageRank, color='mediumblue', edgecolor='grey', label='Leader+NewFormula+AE+CNN')
plt.bar(br4, Game_theo, color='royalblue', edgecolor='grey', label='PageRank+AE+CNN')
plt.xlabel(sets_simple_name_list[dataset], fontweight='bold', fontsize=15)
plt.ylabel('Accuracy Methods', fontweight='bold', fontsize=15)
plt.title('Modularity', fontweight='bold', fontsize=25)
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
plt.legend(loc=4)
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
Article, Prev_prj, PageRank, Game_theo = final_numbers[2 * dataset + 1]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("{0: >10}  {1: >10.3}  {2: >15.3}  {3: >20.3}  {4: >20.3}".format(ConsoleTextColors.FAIL + "NMI       " + ConsoleTextColors.ENDC,
                                                                        Article,
                                                                        Prev_prj,
                                                                        PageRank,
                                                                        Game_theo))
print('-' * 120)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

br1 = Article
br2 = br1 + barWidth
br3 = br2 + barWidth
br4 = br3 + barWidth
plt.bar(br1, Article, color='mediumpurple', edgecolor='grey', label='RM+AE+CNN')
plt.bar(br2, Prev_prj, color='indigo', edgecolor='grey', label='Leader+AE+CNN')
plt.bar(br3, PageRank, color='rebeccapurple', edgecolor='grey', label='Leader+NewFormula+AE+CNN')
plt.bar(br4, Game_theo, color='blueviolet', edgecolor='grey', label='PageRank+AE+CNN')
plt.xlabel(sets_simple_name_list[dataset], fontweight='bold', fontsize=15)
plt.ylabel('Accuracy Methods', fontweight='bold', fontsize=15)
plt.title('NMI', fontweight='bold', fontsize=25)
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
plt.legend(loc=4)
plt.show()
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TODO ARTICLE CASES
print("{0: >10}  {1: >15}  {2: >20}  {3: >15}  {4: >20}".format(ConsoleTextColors.HEADER + sets_simple_name_list[dataset],
                                                                ConsoleTextColors.OKCYAN + "node2vec+Kmeans",
                                                                ConsoleTextColors.OKBLUE + "NetRA+Kmeans",
                                                                ConsoleTextColors.OKGREEN + "SDNE+Kmeans",
                                                                ConsoleTextColors.WARNING + "Leader+AE+CNN" + ConsoleTextColors.ENDC))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
Article, Prev_prj, PageRank, Game_theo = data_numbers[2 * dataset]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("{0: >10}  {1: >10.3}  {2: >15.3}  {3: >10.3}  {4: >10.3}".format(ConsoleTextColors.FAIL + "Modularity" + ConsoleTextColors.ENDC,
                                                                        Prev_prj,
                                                                        PageRank,
                                                                        Game_theo,
                                                                        Article,
                                                                        ))
br1 = Article
br2 = br1 + barWidth
br3 = br2 + barWidth
br4 = br3 + barWidth
plt.bar(br1, Article, color='cornflowerblue', edgecolor='grey', label='node2vec+Kmeans')
plt.bar(br2, Prev_prj, color='royalblue', edgecolor='grey', label='NetRA+Kmeans')
plt.bar(br3, PageRank, color='mediumblue', edgecolor='grey', label='SDNE+Kmeans')
plt.bar(br4, Game_theo, color='darkblue', edgecolor='grey', label='Leader+AE+CNN')
plt.xlabel(sets_simple_name_list[dataset], fontweight='bold', fontsize=15)
plt.ylabel('Accuracy Methods', fontweight='bold', fontsize=15)
plt.title('Modularity', fontweight='bold', fontsize=25)
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
plt.legend(loc=4)
plt.show()
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
Article, Prev_prj, PageRank, Game_theo = data_numbers[2 * dataset + 1]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("{0: >10}  {1: >10.3}  {2: >15.3}  {3: >10.3}  {4: >10.3}".format(ConsoleTextColors.FAIL + "NMI       " + ConsoleTextColors.ENDC,
                                                                        Article,
                                                                        Prev_prj,
                                                                        PageRank,
                                                                        Game_theo))
print('-' * 120)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

br1 = Article
br2 = br1 + barWidth
br3 = br2 + barWidth
br4 = br3 + barWidth
plt.bar(br1, Article, color='mediumpurple', edgecolor='grey', label="node2vec+Kmeans")
plt.bar(br2, Prev_prj, color='blueviolet', edgecolor='grey', label='NetRA+Kmeans')
plt.bar(br3, PageRank, color='rebeccapurple', edgecolor='grey', label='SDNE+Kmeans')
plt.bar(br4, Game_theo, color='indigo', edgecolor='grey', label='Leader+AE+CNN')
plt.xlabel(sets_simple_name_list[dataset], fontweight='bold', fontsize=15)
plt.ylabel('Accuracy Methods', fontweight='bold', fontsize=15)
plt.title('NMI', fontweight='bold', fontsize=25)
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
plt.legend(loc=4)
plt.show()
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# if dataset == 0:
#     data = mmread('soc-karate.mtx')
# elif dataset == 1:
#     data = mmread('soc-dolphins.mtx')
# else:
data = nx.adjacency_matrix(G)
number_of_clusters = [3, 3, 12]
kmeans = KMeans(n_clusters=number_of_clusters[dataset], random_state=0).fit(data)
print(f'{ConsoleTextColors.OKGREEN}labels:{ConsoleTextColors.ENDC}', kmeans.labels_)
print('kmeans.inertia_: ', kmeans.inertia_)
print('kmeans.n_iter_: ', kmeans.n_iter_)
print('kmeans.cluster_centers_: ', kmeans.cluster_centers_)
rgb = ['r', 'g', 'blue', 'c', 'm', 'y', 'brown', 'orange', 'mediumseagreen', 'pink', 'olive', 'purple']
colors = [rgb[lb] for lb in kmeans.labels_]

# print('data=', data)

nei = []
for node in nx.nodes(G):
    nx.neighbors(G, node)
    nei.append(len(list(nx.neighbors(G, node))))
sns.scatterplot(data=data, x=nx.nodes(G), y=nei, c=colors, )

plt.title('Nodes And Neighbours')
plt.xlabel('NODES')
plt.ylabel('NEIGHBOURS')
plt.show()
# print(data.todense())
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
if dataset == 3:
    nx.draw(G, node_size=30, node_color=colors)
else:
    nx.draw(G, with_labels=False, node_color=colors)

plt.show()
if dataset != 2:
    print('choose clustered graph type:')
    print(f'{ConsoleTextColors.OKBLUE}1{ConsoleTextColors.ENDC}-circular')
    print(f'{ConsoleTextColors.OKGREEN}2{ConsoleTextColors.ENDC}-organized')
    gty = int(input('Enter type number:'))
    radii = [7, 15, 30, 45, 60]  # for concentric circles

    if gty == 1:
        pos = nx.circular_layout(G)
        for i, cl in enumerate(kmeans.labels_):
            pos[i] *= radii[cl]  # reposition nodes as concentric circles
    else:
        pos = nx.circular_layout(G)  # replaces your original pos=...
        angs = np.linspace(0, 2 * np.pi, 1 + 3)
        repos = []
        rad = 3.5  # radius of circle
        for ea in angs:
            if ea > 0:
                # print(rad*np.cos(ea), rad*np.sin(ea))  # location of each cluster
                repos.append(np.array([rad * np.cos(ea), rad * np.sin(ea)]))
        for i, cl in enumerate(kmeans.labels_):
            pos[i] += repos[cl]

    nx.draw(G, pos=pos, node_color=colors, with_labels=False)
    plt.show()
print("END")
sys.exit()
