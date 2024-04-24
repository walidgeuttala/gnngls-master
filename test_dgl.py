import networkx as nx
import dgl

# Step 1: Convert NetworkX graph to DGL graph
def nx_to_dgl(nx_graph):
    dgl_graph = dgl.from_networkx(nx_graph, edge_attrs=None)
    return dgl_graph

# Step 2: Create line graph from DGL graph
def create_line_graph(dgl_graph):
    line_graph = dgl.line_graph(dgl_graph, backtracking=False)
    return line_graph

# Step 3: Add different types of edges
def add_edge_types(line_graph, original_graph):
    num_nodes = original_graph.number_of_nodes()
    num_edges = original_graph.number_of_edges()
    
    for i in range(num_edges):
        # source-target edge
        line_graph.add_edges(i, num_edges + i)
        # target-source edge
        line_graph.add_edges(num_edges + i, i)
        # source-source edge
        line_graph.add_edges(i, i)
        # target-target edge
        line_graph.add_edges(num_edges + i, num_edges + i)
    return line_graph

# Example usage:
# Assuming you have a directed NetworkX graph named nx_graph
nx_graph = nx.complete_graph(3).to_directed()  # Example directed complete graph
dgl_graph = nx_to_dgl(nx_graph)
print(dgl_graph.edges())
for edge in zip(*dgl_graph.edges()):
    print(edge)
    