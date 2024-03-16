#import concorde.tsp as concorde
import lkh
import networkx as nx
import numpy as np
import tsplib95
from matplotlib import colors


def tour_to_edge_attribute(G, tour):
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges
    return in_tour


def tour_cost(G, tour, weight='weight'):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += G.edges[e][weight]
    return c


def is_equivalent_tour(tour_a, tour_b):
    if tour_a == tour_b[::-1]:
        return True
    if tour_a == tour_b:
        return True
    return False


def is_valid_tour(G, tour):
    if tour[0] != 0:
        return False
    if tour[-1] != 0:
        return False
    for n in G.nodes:
        c = tour.count(n)
        if n == 0:
            if c != 2:
                return False
        elif c != 1:
            return False
    return True


# def optimal_tour(G, scale=1e3):
#     coords = scale * np.vstack([G.nodes[n]['pos'] for n in sorted(G.nodes)])
#     solver = concorde.TSPSolver.from_data(coords[:, 0], coords[:, 1], norm='EUC_2D')
#     solution = solver.solve()
#     tour = solution.tour.tolist() + [0]
#     return tour


def optimal_cost(G, weight='weight'):
    c = 0
    for e in G.edges:
        if G.edges[e]['in_solution']:
            c += G.edges[e][weight]
    return c


def get_adj_matrix_string(G):
    # Get the lower triangular adjacency matrix with diagonal
    adj_matrix = nx.to_numpy_array(G).astype(int)
    n = adj_matrix.shape[0]
    ans = '''NAME: ATSP
    COMMENT: 64-city problem
    TYPE: ATSP
    DIMENSION: 64
    EDGE_WEIGHT_TYPE: EXPLICIT
    EDGE_WEIGHT_FORMAT: FULL_MATRIX
    EDGE_WEIGHT_SECTION: 
    '''
    for i in range(n):
        # Iterate over columns up to the diagonal
        for j in range(n):
            ans += str(adj_matrix[i][j]) + " "
        ans += "\n"
    # Add EOF
    # adj_matrix_string += "EOF"
    
    return ans.strip()


def fixed_edge_tour(G, e, lkh_path='../LKH-3.0.9/LKH'):
    string = get_adj_matrix_string(G)
    problem = tsplib95.loaders.parse(string)
    problem.fixed_edges = [[n + 1 for n in e]]

    solution = lkh.solve(lkh_path, problem=problem)
    tour = [n - 1 for n in solution[0]] + [0]
    return tour


def plot_edge_attribute(G, attr, ax, **kwargs):
    cmap_colors = np.zeros((100, 4))
    cmap_colors[:, 0] = 1
    cmap_colors[:, 3] = np.linspace(0, 1, 100)
    cmap = colors.ListedColormap(cmap_colors)

    pos = nx.get_node_attributes(G, 'pos')

    nx.draw(G, pos, edge_color=attr.values(), edge_cmap=cmap, ax=ax, **kwargs)
