import copy
import pathlib
import pickle

import dgl
import networkx as nx
import numpy as np
import torch
import torch.utils.data


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
    #G2 = dgl.add_reverse_edges(G2)

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
    return G2

complete_graph = nx.complete_graph(3).to_directed()
print(complete_graph.edges())
g = directed_string_graph(complete_graph)
print(g.num_nodes())
print(g.num_edges())
values = torch.tensor([(1, 2), (2, 3)])
g.ndata['e'] = values
print(g.ndata['e'])


# import torch

# # Example list of tensors
# tensor_list = [torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 2)]

# # Concatenate tensors along the specified dimension (dimension 0 in this case)
# concatenated_tensor = torch.cat(tensor_list, dim=0)

# print(concatenated_tensor)
