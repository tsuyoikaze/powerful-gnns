from util import S2VGraph
import networkx as nx 
import numpy as np 
import torch

def generate_sample(num_samples):
    res = []
    for i in range(num_samples):
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 2)
        x_0 = np.random.randint(0, 50)
        x_1 = np.random.randint(50, 100)
        label = np.random.randint(0, 2)
        if label == 0:
            x_2 = x_0 + x_1
        else:
            x_2 = x_0 - x_1
        node_features = np.array([[x_0], [x_1], [x_2]])
        graph = S2VGraph(g, label, node_features = node_features)
        res.append(graph)

    for g in res:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        edges = [list(pair) for pair in g.g.edges()]

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)
    return res
