import networkx as nx

G = nx.karate_club_graph()

for n in G.nodes:
    print("n=", n)
    nn = list(nx.neighbors(G, n))
    print('len=', len(nn))
    print(nn)
