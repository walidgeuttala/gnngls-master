#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import multiprocessing as mp
import pathlib
import uuid

import networkx as nx
import numpy as np

import gnngls
from gnngls import datasets

import linecache

def prepare_instance(G):
    datasets.set_features(G)
    datasets.set_labels(G)
    return G

def get_solved_instances(n_nodes, n_instances):
    for _ in range(n_instances):
        G = nx.Graph()

        coords = np.random.random((n_nodes, 2))
        for n, p in enumerate(coords):
            G.add_node(n, pos=p)

        for i, j in itertools.combinations(G.nodes, 2):
            w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos'])
            G.add_edge(i, j, weight=w)

        opt_solution = gnngls.optimal_tour(G)
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G


def get_solved_instances2(n_nodes, n_instances, all_instances):
    #all_instances = './tsplib95_10000_instances_64_node/all_instances_lower_triangle_tour_cost.txt'
    # Open the file in read mode
   
    for i in range(n_instances):
        line = linecache.getline(all_instances, i+1).strip()
        G = nx.Graph()
        adj, opt_solution, cost = line.split(',')
        adj = adj.split(' ')

        G.add_nodes_from(range(n_nodes))
        opt_solution = [int(x) for x in opt_solution.split()]
       
        # Add the edges for the DiGraph and be sure that does not have self loops in the node
        for j in range(n_nodes):
            for k in range(n_nodes):
                w = float(adj[j*n_nodes+k])
                if j != k:
                    G.add_edge(j, k, weight=w)
            
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Generate a dataset.')
#     parser.add_argument('n_samples', type=int)
#     parser.add_argument('n_nodes', type=int)
#     parser.add_argument('input_file', type=str)
#     parser.add_argument('output_dir', type=pathlib.Path)
#     args = parser.parse_args()

#     if args.output_dir.exists():
#         raise Exception(f'Output directory {args.output_dir} exists.')
#     else:
#         args.output_dir.mkdir()

#     pool = mp.Pool(processes=None)
#     instance_gen = get_solved_instances2(args.n_nodes, args.n_samples, args.input_file)
#     for G in pool.imap_unordered(prepare_instance, instance_gen):
#         nx.write_gpickle(G, args.output_dir / f'{uuid.uuid4().hex}.pkl')
#     pool.close()
#     pool.join()


def append_text_to_file(filename, text):
    with open(filename, 'a') as file: file.write(text + '\n')


def atsp_to_tsp():
    value = 64e6
    for i in range(10000):
        line = linecache.getline('../tsplib95_10000_instances_64_node/all_instances_adj_tour_cost.txt', i+2).strip()
        adj, opt_solution, cost = line.split(',')
        cost = float(cost)
        cost -= value
        adj = adj.split(' ')[:-1]
        opt_solution = [int(x) for x in opt_solution.split()]
        adj = np.array(adj, dtype=np.int32).reshape(64, 64)
        adj = gnngls.as_symmetric(adj)
        opt_solution = gnngls.tranfer_tour(opt_solution, 64)
        instance_adj_tour_cost = gnngls.convert_adj_string(adj)+','+" ".join(map(str, opt_solution))+','+str(cost)
        append_text_to_file('../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt', instance_adj_tour_cost)

def adjacency_matrix_to_networkx(adj_matrix):
    return nx.Graph(np.triu(adj_matrix))

def optimal_cost(G, weight='weight'):
    c = 0
    for e in G.edges:
        if G.edges[e]['in_solution']:
            c += G.edges[e][weight]
    return c


atsp_to_tsp()