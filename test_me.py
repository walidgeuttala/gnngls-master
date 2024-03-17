import networkx as nx
import pickle 
def print_graph_features(filename):
    # Read the graph from the pickle file
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    
    # Print the features for the first edge
    first_edge = next(iter(G.edges()))
    print("Features for the first edge:")
    print("Edge:", first_edge)
    print("Edge features:", G.edges[first_edge])

    # Print the features for the first node
    first_node = next(iter(G.nodes()))
    print("\nFeatures for the first node:")
    print("Node:", first_node)
    print("Node features:", G.nodes[first_node])

    # Get the number of edges and nodes
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()

    return num_edges, num_nodes

file_name = '../gnngls/data/tsp100/32dbcaa175434fc08f6a6175a8246aee.pkl'
num_nodes, num_edges = print_graph_features(file_name)
