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
        lG = nx.line_graph(G)
        #lG = lG.to_undirected()
        for n in lG.nodes:
            lG.nodes[n]['e'] = n
        
        # why he add the id number of the edegs in the G graph
        self.G = dgl.from_networkx(lG, node_attrs=['e'])
        

    

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
        H.ndata['regret'] = torch.nn.functional.softmax(H.ndata['regret'], dim=1)

        H.ndata['in_solution'] = torch.tensor(regret, dtype=torch.float32)
        return H
