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
    t2 = torch.zeros(64, 64, dtype=torch.float32)
    cnt = 0
    for i in range(64):
        for j in range(64):
            if i == j:
                continue
            t2[i][j] = t1[cnt]
            cnt += 1
    return t2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('data_path', type=pathlib.Path)
    parser.add_argument('model_path', type=pathlib.Path)
    parser.add_argument('run_dir', type=pathlib.Path)
    parser.add_argument('guides', type=str, nargs='+')
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

            init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')
            
        else:
            init_tour = algorithms.nearest_neighbor(G, 0, weight='weight')








        with open(args.output_path / f"instance{cnt}.txt", "w") as f:
            # Save array1
            f.write("edge_weight:\n")
            np.savetxt(f, edge_weight, fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array2
            f.write("regret:\n")
            np.savetxt(f, add_diag(H.y.cpu()).numpy(), fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array3
            f.write("regret_pred:\n")
            np.savetxt(f, add_diag(y_pred.cpu()).numpy(), fmt="%.8f", delimiter=" ")
            f.write("\n")

            f.write(f"opt_cost: {opt_cost}\n")
            f.write(f"num_iterations: {cnt_ans}\n")
            f.write(f"init_cost: {init_cost}\n")
            f.write(f"best_cost: {best_cost}\n")



        num_nodes = G.number_of_nodes()
        # atsp_edge_weight, _ = nx.attr_matrix(G, 'weight')
        # tsp = TSPExact(atsp_edge_weight)
        # tsp_edge_weight = tsp.cost_matrix
        # tsp_tour = tsp.tranfer_tour(init_tour, num_nodes)
        # tsp_G = nx.Graph(np.triu(tsp_edge_weight))
        value = 1e6 * num_nodes / 2
        init_cost = gnngls.tour_cost(G, init_tour)
        
        best_tour, best_cost, search_progress_i = algorithms.guided_local_search(G, init_tour, init_cost,
                                                                                 t + args.time_limit, weight='weight',
                                                                                 guides=args.guides,
                                                                                 perturbation_moves=args.perturbation_moves,
                                                                                 first_improvement=False, value=0)
        for row in search_progress_i:
            row.update({
                'instance': instance,
                'opt_cost': opt_cost
            })
            search_progress.append(row)
        best_cost += value
        opt_cost  += value
     
        gap = (best_cost / opt_cost - 1) * 100
        gaps.append(gap)
        print(f'best_cost {best_cost} opt_cost {opt_cost}', flush=True)
        print('Avg Gap: {:.4f}'.format(np.mean(gaps)), flush=True)

        
        

    search_progress_df = pd.DataFrame.from_records(search_progress)
    search_progress_df['best_cost'] = search_progress_df.groupby('instance')['cost'].cummin()
    search_progress_df['gap'] = (search_progress_df['best_cost'] / search_progress_df['opt_cost'] - 1) * 100
    search_progress_df['dt'] = search_progress_df['time'] - search_progress_df.groupby('instance')['time'].transform(
        'min')

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{uuid.uuid4().hex}.pkl'
    if not args.run_dir.exists():
        args.run_dir.mkdir()
    search_progress_df.to_pickle(args.run_dir / run_name)
