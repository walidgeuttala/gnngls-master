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


def prepare_instance(G):
    datasets.set_features(G)
    datasets.set_labels(G)
    return G
walid

def get_solved_instances(n_nodes, n_instances):
    for _ in range(n_instances):
        G = nx.Graph()

        coords = np.random.random((n_nodes, 2))
        for n, p in enumerate(coords):
            G.add_node(n, pos=p)

        for i, j in itertools.combinations(G.nodes, 2):
            w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos'])
            G.add_edge(i, j, weight=w)

        opt_solution = gnngls.optimal_tour(G, scale=1e6)
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G

import linecache
def get_solved_instances2(n_nodes, n_instances):
    all_instances = './tsplib95_10000_instances_64_node/all_instances_lower_triangle_tour_cost.txt'
    # Open the file in read mode
   
    for i in range(n_instances):
        line = linecache.getline(all_instances, i+2).strip()
        G = nx.Graph()
        adj, opt_solution, cost = line.split(',')
        adj = adj.split(' ')
        G.add_nodes_from(range(n_nodes))
        opt_solution = [int(x) for x in opt_solution.split()]
        cnt = 0
        for j in range(n_nodes):
            for k in range(j+1):
                w = float(adj[cnt])
                cnt += 1
                if j == k:
                    continue
                G.add_edge(j, k, weight=w)
            
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset.')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('n_nodes', type=int)
    parser.add_argument('dir', type=pathlib.Path)
    args = parser.parse_args()

    if args.dir.exists():
        raise Exception(f'Output directory {args.dir} exists.')
    else:
        args.dir.mkdir()

    pool = mp.Pool(processes=None)
    instance_gen = get_solved_instances2(args.n_nodes, args.n_samples)
    for G in pool.imap_unordered(prepare_instance, instance_gen):
        nx.write_gpickle(G, args.dir / f'{uuid.uuid4().hex}.pkl')
    pool.close()
    pool.join()
