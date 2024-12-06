# netowrks
import networkx as nx
import igraph as ig

# data processing
import pandas as pd
import numpy as np

#some functions to make our lifes easier
import sys
sys.path.append("../")

# viz
#import pylab as plt
import matplotlib.pyplot as plt
import seaborn as sns

# gzip
import gzip
import statistics

from scipy.sparse import diags, csr_matrix, linalg

import time

# Path to file
file_path = 'soc-Epinions1.txt.gz'

# Initialize a directed graph
G = nx.DiGraph()

# Load the edge list into the directed graph
with gzip.open(file_path, 'rt') as f:
    # Skip header lines that start with '#'
    edges = [line.strip().split('\t') for line in f if not line.startswith('#')]
    # Add edges to the graph
    G.add_edges_from((int(src), int(dst)) for src, dst in edges)

def get_core_nodes(graph, threshold):
    """
    Recursively removes nodes with in-degree + out-degree < threshold and returns the core graph.
    
    Parameters:
    - graph: A NetworkX DiGraph (directed graph)
    
    Returns:
    - A NetworkX DiGraph representing the core of the network
    """
    # Create a copy of the graph to avoid modifying the original
    core_graph = graph.copy()
    
    count = 0
    number_of_removed = 0
    while True:
        # Identify nodes with in-degree + out-degree < threshold
        nodes_to_remove = [
            node for node in core_graph.nodes 
            if core_graph.in_degree(node) + core_graph.out_degree(node) < threshold
        ]
        
        # Exit condition - if no nodes meet the condition, stop the process
        if not nodes_to_remove:
            break
        
        # Remove the identified nodes
        core_graph.remove_nodes_from(nodes_to_remove)
        number_of_removed += len(nodes_to_remove)
        print(f'Iteration {count}: removed {len(nodes_to_remove)} nodes')
        count += 1
    print(f'Executed {count} iterations, removed {number_of_removed} nodes')
    return core_graph

# Get the core graph
threshold = 4
core_G = get_core_nodes(G, threshold)
# Print the number of nodes and edges in the original and core graphs
print(f"Original Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Core Graph: {core_G.number_of_nodes()} nodes, {core_G.number_of_edges()} edges")

def compute_graph_matrices(core_graph):
    """
    Computes the adjacency, degree, and Laplacian matrices for a core graph.

    Parameters:
    - core_graph: A NetworkX DiGraph (the core graph)

    Returns:
    - adjacency_matrix: A NumPy array (core_graph size x core_graph size)
    - degree_matrix: A sparse diagonal matrix (scipy.sparse)
    - laplacian_matrix: A sparse Laplacian matrix (scipy.sparse)
    - core_nodes: A list of node labels in the order they appear in the matrix
    """
    # Get the nodes of the core graph sorted by their original labels
    core_nodes = sorted(core_graph.nodes)
    node_mapping = {node: idx for idx, node in enumerate(core_nodes)}
    
    # Initialize adjacency matrix of size (core_graph size x core_graph size)
    size = len(core_nodes)
    adjacency_matrix = np.zeros((size, size))
    
    # Populate the adjacency matrix
    for u, v in core_graph.edges:
        adjacency_matrix[node_mapping[u], node_mapping[v]] = 1  # Directed edge from u to v

    # Compute the degree for each node (sum of rows in the adjacency matrix)
    degrees = np.sum(adjacency_matrix, axis=1)
    
    # Create the degree matrix as a sparse diagonal matrix
    degree_matrix = diags(degrees, offsets=0, format="csr")
    
    # Compute the Laplacian matrix
    laplacian_matrix = degree_matrix - csr_matrix(adjacency_matrix)

    return adjacency_matrix, degree_matrix, laplacian_matrix, core_nodes

adjacency_matrix, degree_matrix, laplacian_matrix, core_nodes = compute_graph_matrices(core_G)

# print an example
print(laplacian_matrix.shape)

# Number of eigenvalues/vectors to compute (example: 6 smallest non-zero eigenvalues)

k = core_G.number_of_nodes()

start_time = time.time()
# Compute eigenvalues and eigenvectors using ARPACK via eigsh
eigenvalues, eigenvectors = linalg.eigsh(laplacian_matrix.toarray(), k=k, which='SM')  # SM: smallest magnitude
elapsed_time = time.time() - start_time

print (elapsed_time)

# Output results
# print("Eigenvalues:", eigenvalues)
print("Eigenvectors (shape):", eigenvectors.shape)


# Save the results to a file
np.savez("eigenvalues_eigenvectors.npz", eigenvalues=eigenvalues, eigenvectors=eigenvectors)

print("Eigenvalues and eigenvectors have been saved to 'eigenvalues_eigenvectors.npz'")