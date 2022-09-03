import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import mmread
import math
import random


def weird_division(n, d):
    try:
        return n / d
    except ZeroDivisionError:
        return 0


def listCoaSet(CoaSet):
    CoaList = []
    for Coa in CoaSet:
        CoaList.append(list(Coa))
        # print(Coa)

    return CoaList


def removeFromList(List, Item):
    if Item not in List: return
    List.remove(Item)


dataset = False
print("*** Welcome to this program ***")

# dataset = mmread('soc-karate.mtx')
dataset = mmread('../soc-karate.mtx')
# dataset = mmread('../soc-dolphins.mtx')
G = nx.from_scipy_sparse_array(dataset)

Nodes = nx.nodes(G)
NodeCount = nx.number_of_nodes(G)
Edges = nx.edges(G)
EdgeCount = nx.number_of_edges(G)

alpha = weird_division(1, math.sqrt(EdgeCount))
beta = 1
TwoBE = 2 * beta * EdgeCount
sqrt_TwoBE = math.sqrt(TwoBE)


def get_coa_set_map(graph, numberOfNodes=1, isRandom=False):
    keys = list(range(0, nx.number_of_nodes(graph)))
    # print(keys)
    if (isRandom == True):
        random.shuffle(keys)
    # Get a list of node lists splited by numberOfNodes from graph nodes kyes
    output = [keys[i:i + numberOfNodes] for i in range(0, len(keys), numberOfNodes)]

    # Build an array of subgraphs 
    coa_set = []
    for node_list in output:
        subgraph = graph.subgraph(node_list)
        coa_set.append(subgraph)

    return coa_set


k = 0
CoaSets = []
coa_set0 = get_coa_set_map(G, 3, True)
CoaSets.append(coa_set0)  # Add All Nodes To CoaSet0


def sum_of_degree(S):
    if (S == False):
        return 0
    nodes = list(S.nodes)
    dS = 0
    for node in nodes:
        dS = dS + S.degree(node)
    return dS


def calculate_ds(S1, S2=False):
    dS1 = sum_of_degree(S1)
    # print(f"d(S1) => {dS1} - S1 => {list(S1.nodes)}")

    dS2 = 0
    if (S2 != False):
        dS2 = sum_of_degree(S2)
        # print(f"d(S2) => {dS2} - S2 => {list(S2.nodes)}")

    return dS1 + dS2


def calculate_es(S1, S2=False):
    eS = nx.number_of_edges(S1)

    if (S2 != False):
        eS = eS + (nx.number_of_edges(S2))

    return eS


def calculate_utility(S1, S2=False, UseCompose=False):
    global alpha, TwoBE

    eS = calculate_es(S1, S2)
    dS = calculate_ds(S1, S2)

    if S2 != False & UseCompose != False:
        compose_S1S2 = nx.compose(S1, S2)
        eS = calculate_es(compose_S1S2)
        dS = calculate_ds(compose_S1S2)

    return (weird_division((eS * 2), dS)) - (alpha * (pow(weird_division(dS, (TwoBE)), 2)))


def getCoaSps(CoaSet, Coa2=False):
    best_utility = False
    best_coa = False
    # print(calculate_utility(CoaSet))
    for Coa in CoaSet:
        utility = calculate_utility(Coa, Coa2)
        # print(F"Coa Nodes: {Coa.nodes} Utility: {utility}")
        if (best_utility == False or utility > best_utility):
            best_utility = utility
            best_coa = Coa
    # print(F"Best Coa Nodes: {best_coa.nodes} Best Utility: {best_utility}")
    return best_coa


def getCooCaSet_CooSps(CoaSet, CooSps):
    global G
    CooCaSet = []
    for coa in CoaSet:
        for coa_node in list(coa.nodes):
            for coo_sps_node in list(CooSps.nodes):
                try:
                    has_edge = G.has_edge(coa_node, coo_sps_node)
                    if has_edge:
                        if coa not in CooCaSet:
                            CooCaSet.append(coa)
                except:
                    print("An exception occurred")
                    continue

    return CooCaSet
    # for Coa in CoaSet:


def do():
    print('\n\n**** Start Of Do ****')
    global k, CoaSets, sqrt_TwoBE
    # print(k,(k-1),CoaSets[(k-1)])
    CoaSetK_1 = CoaSets[(k - 1)].copy() if (k - 1) >= 0 & (k - 1) < len(CoaSets) else []
    # print(CoaSetK_1)

    CoaSetMap = CoaSets[k].copy()
    k = k + 1
    newCoaSet = []
    while CoaSetMap != []:
        print("-------------------------------------------------------")
        print(f"CoaSetMap => {listCoaSet(CoaSetMap)}")
        print(f"CoaSet(K-1) => {listCoaSet(CoaSetK_1)}")
        CooSps = getCoaSps(CoaSetMap)
        # print(f"CooSps => {list(CooSps)}")
        removeFromList(CoaSetMap, CooSps)

        CooCaSet_CooSps = getCooCaSet_CooSps(CoaSetK_1, CooSps)
        print(f"CooCaSet(CooSps) => {listCoaSet(CooCaSet_CooSps)}")

        while CooCaSet_CooSps != []:
            print("*******************************************************")
            print(f"  CooCaSet(CooSps) => {listCoaSet(CooCaSet_CooSps)}")
            print(f"            CooSps => {list(CooSps)}")
            CooCas_Star = getCoaSps(CooCaSet_CooSps, CooSps)
            print(f"           CooCas* => {list(CooCas_Star)}")

            # CooSps_CooCas_Star = nx.compose(CooSps,CooCas_Star)
            # print(f"    CooSps_CooCas* => {list(CooSps_CooCas_Star)}")

            # print(f"Utility1 => {calculate_utility(CooSps,CooCas_Star)} Utility2 => {calculate_utility(CooSps_CooCas_Star)}")
            # CooSps_CooCas_Star_Es = calculate_es(CooSps_CooCas_Star)
            CooSps_CooCas_Star_Es = calculate_es(CooSps, CooCas_Star)
            # CooSps_CooCas_Star_Vs = calculate_utility(CooSps_CooCas_Star)
            CooSps_Vs = calculate_utility(CooSps)
            CooCas_Star_Vs = calculate_utility(CooCas_Star)
            CooSps_CooCas_Star_Vs = CooSps_Vs + CooCas_Star_Vs

            condition1 = CooSps_CooCas_Star_Es < sqrt_TwoBE
            condition2 = CooSps_CooCas_Star_Vs > CooSps_Vs
            condition3 = CooSps_CooCas_Star_Vs > CooCas_Star_Vs

            if (condition1 & condition2 & condition3):
                print("%%%%%%%%%%%%%%%%%%%%% Conditions %%%%%%%%%%%%%%%%%%%%%")

                CooCaSet_CooCas_Star = getCooCaSet_CooSps(CoaSetK_1, CooCas_Star)
                removeFromList(CooCaSet_CooCas_Star, CooSps)
                # CooCaSet_CooCas_Star.remove(CooSps)

                print(f" CooCaSet(CooCas*)-[CooSps] => {listCoaSet(CooCaSet_CooCas_Star)}")

                # print(f" e(CooSps+CooCas*) => {CooSps_CooCas_Star_Es}")
                # print(f"            √2*B*E => {sqrt_TwoBE}")
                # print(f" v(CooSps+CooCas*) => {CooSps_CooCas_Star_Vs}")
                # print(f"         v(CooSps) => {CooSps_Vs}")
                # print(f"        v(CooCas*) => {CooCas_Star_Vs}")

                # print("=================== Conditions ===================")
                # print(f"     e(CooSps+CooCas*) < √2*B*E     => {condition1}")
                # print(f"     v(CooSps+CooCas*) > v(CooSps)  => {condition2}")
                # print(f"     v(CooSps+CooCas*) < v(CooCas*) => {condition3}")
                # print("==================================================")

                print(f"       CooSps(Before merge) => {list(CooSps)}")
                CooSps = nx.compose(CooSps, CooCas_Star)
                print(f"        CooSps(After merge) => {list(CooSps)}")

                removeFromList(CoaSetMap, CooCas_Star)

                removeFromList(CooCaSet_CooSps, CooCas_Star)

                # CooCaSet_CooSps = CooCaSet_CooSps + CooCaSet_CooCas_Star

                print(f" CooCaSet(CooSps) => {listCoaSet(CooCaSet_CooSps)}")

                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            else:
                removeFromList(CooCaSet_CooSps, CooCas_Star)
                # CooCaSet_CooSps.remove(CooCas_Star)

        newCoaSet.append(CooSps)
    CoaSets.append(newCoaSet)
    print(f"NewCoaSetMap => {listCoaSet(newCoaSet)}")
    print('\n\n**** End Of DO ****')


while True:
    do()
    # if CoaSets[k] == CoaSets[k-1]:
    if k == 2:
        print(f"===================== is same    {CoaSets[k] == CoaSets[k - 1]}")
        print(f'\nCoaSets[{k}] Is equal with CoaSets[{k - 1}] so break from loop...')

        print('\nList Of Coalition Sets:')
        for i in range(k + 1):
            print(f'\n\nCoaSet[{i}]: {listCoaSet(CoaSets[k])}')
        break

# print(list(nx.neighbors(G,20)))
# print(a)
# nx.draw(G, with_labels=True)
# plt.show()
