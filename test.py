#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import json
import pathlib
import time
import uuid

import networkx as nx
import numpy as np
import pandas as pd
import torch
#import tqdm.auto as tqdm

import gnngls
from gnngls import algorithms, models, datasets
from atps_to_tsp import TSPExact

def add_diag(t1):
    n = 64
    t2 = torch.zeros(n, n, dtype=torch.float32)
    cnt = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t2[i][j] = t1[cnt]
            cnt += 1
    return t2

def correlation_matrix(tensor1, tensor2):
    
    # Flatten tensors into 1D arrays
    flat_tensor1 = tensor1.flatten().numpy()
    flat_tensor2 = tensor2.flatten().numpy()

    # Concatenate flattened tensors along the second axis

    # Compute the correlation matrix
    corr_matrix = np.corrcoef(flat_tensor1, flat_tensor2)[0, 1]
    
    return corr_matrix

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity
def tsp_to_atsp_instance(G1):
    num_nodes = G1.number_of_nodes() // 2
    G2 = nx.DiGraph()
    G2.add_nodes_from(range(num_nodes))
    G2.add_edges_from([(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v])

    first_edge = list(G1.edges)[0]

    # Get the attribute names of the first edge
    attribute_names = G1[first_edge[0]][first_edge[1]].keys()
    attribute_names_list = list(attribute_names)
    for attribute_name in attribute_names_list:
        attribute, _ = nx.attr_matrix(G1, attribute_name)
        attribute = attribute[num_nodes:, :num_nodes]
        for u, v in G2.edges():
            G2[u][v][attribute_name] = attribute[u, v]
    
    return G2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('data_path', type=pathlib.Path)
    parser.add_argument('model_path', type=pathlib.Path)
    parser.add_argument('run_dir', type=pathlib.Path)
    parser.add_argument('guides', type=str, nargs='+')
    parser.add_argument('output_path', type=pathlib.Path)
    parser.add_argument('--time_limit', type=float, default=10.)
    parser.add_argument('--perturbation_moves', type=int, default=20)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()
    params = json.load(open(args.model_path.parent / 'params.json'))
    if 'efeat_drop_idx' in params:
        test_set = datasets.TSPDataset(args.data_path, feat_drop_idx=params['efeat_drop_idx'])
    else:
        test_set = datasets.TSPDataset(args.data_path)

    if 'regret_pred' in args.guides:
        device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        print('device =', device)

        _, feat_dim = test_set[0].ndata['weight'].shape

        model = models.EdgePropertyPredictionModel(
            feat_dim,
            params['embed_dim'],
            1,
            params['n_layers'],
            n_heads=params['n_heads']
        ).to(device)

        checkpoint = torch.load(args.model_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    #pbar = tqdm.tqdm(test_set.instances)
    gaps = []
    search_progress = []
    cnt = 0
    corr_all = 0.
    for instance in test_set.instances:
        G = nx.read_gpickle(test_set.root_dir / instance)

        opt_cost = gnngls.optimal_cost(G, weight='weight')

        t = time.time()
        search_progress.append({
            'instance': instance,
            'time': t,
            'opt_cost': opt_cost
        })

        if 'regret_pred' in args.guides:
            H = test_set.get_scaled_features(G).to(device)
            x = H.ndata['weight']
            y = H.ndata['regret']
            with torch.no_grad():
                y_pred = model(H, x)
            regret_pred = test_set.scalers['regret'].inverse_transform(y_pred.cpu().numpy())
            es = H.ndata['e'].cpu().numpy()
            for e, regret_pred_i in zip(es, regret_pred):
                G.edges[e]['regret_pred'] = np.maximum(regret_pred_i.item(), 0)
            G = tsp_to_atsp_instance(G)
            init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')
        
        init_cost = gnngls.tour_cost(G, init_tour)



        edge_weight, _ = nx.attr_matrix(G, 'weight')
        corr = correlation_matrix(y_pred.cpu(),H.ndata['regret'].cpu())
        corr_all += corr

        with open(args.output_path / f"instance{cnt}.txt", "w") as f:
            # Save array1
            f.write("edge_weight:\n")
            np.savetxt(f, edge_weight, fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array2
            f.write("regret:\n")
            np.savetxt(f, add_diag(H.ndata['regret'].cpu()).numpy(), fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array3
            f.write("regret_pred:\n")
            np.savetxt(f, add_diag(y_pred.cpu()).numpy(), fmt="%.8f", delimiter=" ")
            f.write("\n")

            f.write(f"opt_cost: {opt_cost}\n")
            f.write(f"init_cost: {init_cost}\n")
            f.write(f"correlation: {corr}\n")
            

        cnt += 1

        # num_nodes = G.number_of_nodes()
        # # atsp_edge_weight, _ = nx.attr_matrix(G, 'weight')
        # # tsp = TSPExact(atsp_edge_weight)
        # # tsp_edge_weight = tsp.cost_matrix
        # # tsp_tour = tsp.tranfer_tour(init_tour, num_nodes)
        # # tsp_G = nx.Graph(np.triu(tsp_edge_weight))
        # value = 1e6 * num_nodes / 2
        # init_cost = gnngls.tour_cost(G, init_tour)
        
        # best_tour, best_cost, search_progress_i = algorithms.guided_local_search(G, init_tour, init_cost,
        #                                                                          t + args.time_limit, weight='weight',
        #                                                                          guides=args.guides,
        #                                                                          perturbation_moves=args.perturbation_moves,
        #                                                                          first_improvement=False, value=0)
        # for row in search_progress_i:
        #     row.update({
        #         'instance': instance,
        #         'opt_cost': opt_cost
        #     })
        #     search_progress.append(row)
        # best_cost += value
        # opt_cost  += value
     
        # gap = (best_cost / opt_cost - 1) * 100
        # gaps.append(gap)
        # print(f'best_cost {best_cost} opt_cost {opt_cost}', flush=True)
        # print('Avg Gap: {:.4f}'.format(np.mean(gaps)), flush=True)

        print(corr_all/cnt)
        

    # search_progress_df = pd.DataFrame.from_records(search_progress)
    # search_progress_df['best_cost'] = search_progress_df.groupby('instance')['cost'].cummin()
    # search_progress_df['gap'] = (search_progress_df['best_cost'] / search_progress_df['opt_cost'] - 1) * 100
    # search_progress_df['dt'] = search_progress_df['time'] - search_progress_df.groupby('instance')['time'].transform(
    #     'min')

    # timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    # run_name = f'{timestamp}_{uuid.uuid4().hex}.pkl'
    # if not args.run_dir.exists():
    #     args.run_dir.mkdir()
    # search_progress_df.to_pickle(args.run_dir / run_name)
