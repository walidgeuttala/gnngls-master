import networkx as nx

# Create a complete directed graph
G = nx.complete_graph(3, create_using=nx.DiGraph())

# Add weights to edges
weights = [1, 2, 3, 4, 5, 6]
for i, (u, v) in enumerate(G.edges()):
    G.edges[u, v]['weight'] = weights[i]

# Print edges with weights
for u, v, w in G.edges(data='weight'):
    print(f"Edge ({u}, {v}) has weight {w}")
