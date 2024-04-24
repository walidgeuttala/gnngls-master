import copy
import pathlib
import pickle

import dgl
import networkx as nx
import numpy as np
import torch
import torch.utils.data

from . import tour_cost, fixed_edge_tour, optimal_cost as get_optimal_cost


def set_features(G):
    for e in G.edges:
        G.edges[e]['features'] = np.array([
            G.edges[e]['weight'],
        ], dtype=np.float32)

def set_labels(G):
    optimal_cost = get_optimal_cost(G)
    if optimal_cost == 0:
        optimal_cost = 1e-6
    if optimal_cost < 0:
        value = -1.
    else:
        value = 1.
    for e in G.edges:
        regret = 0.

        if not G.edges[e]['in_solution']: 

            tour = fixed_edge_tour(G, e)
            cost = tour_cost(G, tour)
            regret = (cost - optimal_cost) / optimal_cost * value
            
        G.edges[e]['regret'] = regret

def set_labels2(G):
    optimal_cost = get_optimal_cost(G)
    for e in G.edges:
        regret = 0.
        G.edges[e]['regret'] = regret

# def string_graph(G1):
#     G2 = nx.Graph()
#     num_nodes = G1.number_of_edges()
#     nodes = range(0, num_nodes*(num_nodes+1)/2)
#     G2.add_nodes_from(nodes)
#     for edge in G1.edges():
#         s, t = edge
#         for neighbor in range(0, num_nodes):
#             if neighbor == t:
#                 pass
#             elif neighbor == s:
#                 pass
#             else:
#                 pass

def directed_string_graph(G1):
    n = G1.number_of_nodes()
    m = n*(n-1)

    i, j = 0, 1
    ss = []
    st = []
    ts = []
    tt = []
    pp = []
    edge_id = dict()
    # features = []
    # regret = []
    # in_solution = []
    for idx, edge in enumerate(G1.edges()):
        edge_id[edge] = idx
    #     features.append(G1.edges[edge]['weight'])
    #     regret.append(G1.edges[edge]['regret'])
    #     in_solution.append(G1.edges[edge]['in_solution'])

    # features = np.vstack(features)
    # regret = np.vstack(regret)
    # in_solution = np.vstack(in_solution)
    print(edge_id)
    set_list = set()
    for idx in range(m):
        # parallel
        if (i, j, j, i) not in set_list:
            set_list.add((i, j, j, i))
            set_list.add((j, i, i, j))
            pp.append((edge_id[(i, j)], edge_id[(j, i)]))
        # src to src
        for v in range(n):
            if v != i and v != j and (i, j, i, v) not in set_list:
                set_list.add((i, j, i, v))
                set_list.add((i, v, i, j))
                ss.append((edge_id[(i, j)], edge_id[(i, v)]))
        # src to target
        for v in range(n):
            if v != i and v != j and (i, j, v, i) not in set_list:
                set_list.add((i, j, v, i))
                set_list.add((v, i, i, j))
                st.append((edge_id[(i, j)], edge_id[(v, i)]))
        # target to src
        for v in range(n):
            if v != i and v != j and (i, j, j, v) not in set_list:
                set_list.add((i, j, j, v))
                set_list.add((j, v, i, j))
                ts.append((edge_id[(i, j)], edge_id[(j, v)]))
        # target to target
        for v in range(n):
            if v != i and v != j and (i, j, v, j) not in set_list:
                set_list.add((i, j, v, j))
                set_list.add((v, j, i, j))
                tt.append((edge_id[(i, j)], edge_id[(v, j)]))
        

        j += 1
        if i == j:
            j += 1
        if j == n:
            j = 0
            i += 1
    edge_types = {('node', 'ss', 'node'): ss,
              ('node', 'st', 'node'): st,
              ('node', 'ts', 'node'): ts,
              ('node', 'tt', 'node'): tt,
              ('node', 'pp', 'node'): pp}
    
    G2 = dgl.heterograph(edge_types)
    G2 = G2.add_reverse_edges(G2)

    # G2.ndata['weight'] = torch.tensor(features, dtype=torch.float32)
    # G2.ndata['regret'] = torch.tensor(regret, dtype=torch.float32)
    # G2.ndata['in_solution'] = torch.tensor(in_solution, dtype=torch.float32)

    # Print the number of nodes and edges
    print("Number of nodes:", G2.number_of_nodes())
    print("Number of edges:", G2.number_of_edges())

    # Print node and edge features
    print("Node features:")
    print(G2.ndata)
    print("Edge features:")
    print(G2.edata)

    # Print graph structure
    print("Graph structure:")
    print(G2)
    for etype in G2.canonical_etypes:
        print(etype)
        print(G2.adj(etype=etype))

class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, instances_file, scalers_file=None, feat_drop_idx=[]):
        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)
        self.root_dir = instances_file.parent

        self.instances = sorted([line.strip() for line in open(instances_file)])

        if scalers_file is None:
            scalers_file = self.root_dir / 'scalers.pkl'
        scalers = pickle.load(open(scalers_file, 'rb'))
        if 'edges' in scalers: # for backward compatability
            self.scalers = scalers['edges']
        else:
            self.scalers = scalers

        self.feat_drop_idx = feat_drop_idx

        # only works for homogenous datasets
        G = nx.read_gpickle(self.root_dir / self.instances[0])
        directed_string_graph(G)
        

    

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        G = nx.read_gpickle(self.root_dir / self.instances[i])
        H = self.get_scaled_features(G)
        return H

    def get_scaled_features(self, G):
        features = []
        regret = []
        in_solution = []
        for i in range(self.G.number_of_nodes()):
            e = tuple(self.G.ndata['e'][i].numpy())  # corresponding edge

            features.append(G.edges[e]['weight'])
            regret.append(G.edges[e]['regret'])
            in_solution.append(G.edges[e]['in_solution'])

        features = np.vstack(features)
        features_transformed = self.scalers['weight'].transform(features)
        features_transformed = np.delete(features_transformed, self.feat_drop_idx, axis=1)
        regret = np.vstack(regret)
        regret_transformed = self.scalers['regret'].transform(regret)
        in_solution = np.vstack(in_solution)

        H = copy.deepcopy(self.G)
        H.ndata['weight'] = torch.tensor(features_transformed, dtype=torch.float32)
        H.ndata['regret'] = torch.tensor(regret_transformed, dtype=torch.float32)

        H.ndata['in_solution'] = torch.tensor(in_solution, dtype=torch.float32)
        return H


