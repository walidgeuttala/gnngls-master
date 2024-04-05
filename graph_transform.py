# Transform Graphs from TSP to ATSP
import os
import pickle
import networkx as nx
import pathlib

def tsp_to_atsp(input_dir, output_dir):
    instances = list(input_dir.glob('instance*.pkl'))
    os.mkdir(output_dir)
    for idx, instance in enumerate(instances):
        output_name = output_dir / f'instance{idx}.pkl'
        with open(input_dir / instance, 'rb') as file:
            G1 = pickle.load(file)
        if G1.number_of_nodes() != 128:
            print('wrong')
            break
        num_nodes = G1.number_of_nodes() // 2
        G2 = nx.DiGraph()
        G2.add_nodes_from(range(num_nodes))
        G2.add_edges_from([(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v])
        # Get the weight
        weight, _ = nx.attr_matrix(G1, 'weight')
        weight = weight[64:, :64]
        # Get the regret
        regret, _ = nx.attr_matrix(G1, 'regret')
        regret = regret[64:, :64]
        for u, v in G2.edges():
            G2[u][v]['weight'] = weight[u, v]
            G2[u][v]['regret'] = regret[u, v]
        with open(output_name, 'wb') as file:
            pickle.dump(G2, file)

tsp_to_atsp(pathlib.Path('../cleaned_data_n5900'), pathlib.Path('../atsp_n5900'))