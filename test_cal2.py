import gnngls
from gnngls import algorithms, datasets
import networkx as nx
import numpy as np
import time
import tqdm.auto as tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

test_set = datasets.TSPDataset("../tsp_n5900/test3.txt")


def calculate_stats(data):
    # Convert the list of lists into a NumPy array
    arr = np.array(data)
    
    # Calculate statistics
    min_val = np.min(arr)
    max_val = np.max(arr)
    avg_val = np.mean(arr)
    q1 = np.percentile(arr, 25)
    q2 = np.percentile(arr, 50)  # Median
    q3 = np.percentile(arr, 75)
    std_dev = np.std(arr)
    
    # Return the statistics as a NumPy array
    stats = np.array([min_val, max_val, avg_val, q1, q2, q3, std_dev])
    
    return stats

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

x_axis = [1, 2, 4, 8, 16, 32]
init_gap_axis = []
mid_gap_axis = []
final_gap_axis = []


for noise in x_axis:
    init_gaps = []
    mid_gaps = []
    final_gaps =[]
    pbar = tqdm.tqdm(test_set.instances)
    dir = "../noisyRegrets/withStandardDeviation0_{:02d}/".format(noise)
    for idx, instance in enumerate(pbar):
        G = nx.read_gpickle(test_set.root_dir / instance)
        G = tsp_to_atsp_instance(G)
        
        opt_cost = gnngls.optimal_cost(G, weight='weight')
        pred_reg = np.loadtxt(dir + f"instance{idx}.txt")
        for i in range(64):
            for j in range(64):
                if i == j:
                    continue
                G[i][j]['regret_pred'] = np.maximum(pred_reg[i][j].item(), 0)
        init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')
        init_cost = gnngls.tour_cost(G, init_tour)
        edge_weight, _ = nx.attr_matrix(G, 'weight')
        t = time.time()
        cur_tour, cur_cost, search_progress, cnt = algorithms.local_search(init_tour, init_cost, edge_weight, False)
        best_tour, best_cost, search_progress_i, cnt_ans = algorithms.guided_local_search(G, init_tour, init_cost,
                                                                                    t + 10., weight='weight',
                                                                                    guides=["regret_pred"],
                                                                                    perturbation_moves=20,
                                                                                    first_improvement=False)

        costy = gnngls.tour_cost(G, best_tour)
        if best_cost != costy:
            print('error',flush=True)
            break

        init_gaps.append((init_cost/ opt_cost - 1) * 100)
        mid_gaps.append((cur_cost/ opt_cost - 1) * 100)
        final_gaps.append((best_cost/ opt_cost - 1) * 100)
    pbar.close()
    init_gap_axis.append(init_gaps)
    mid_gap_axis.append(mid_gaps)
    final_gap_axis.append(final_gaps)

result = [init_gap_axis, mid_gap_axis, final_gap_axis]
np.save('my_array.npy', result)

