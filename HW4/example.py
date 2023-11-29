import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import networkx as nx
import csv

# Generate random points
#np.random.seed(42)
N = 40
points = np.random.rand(N, 2)

# Perform Delaunay triangulation
triangulation = Delaunay(points)

# Create a graph from the Delaunay triangulation
graph = nx.Graph()

# Add nodes and edges to the graph
for simplex in triangulation.simplices:
    graph.add_edge(simplex[0], simplex[1])
    graph.add_edge(simplex[1], simplex[2])
    graph.add_edge(simplex[2], simplex[0])

# Draw the graph
pos = {i: points[i] for i in range(N)}
nx.draw_networkx(graph, pos, with_labels=True, font_weight='bold', node_size=200, node_color='skyblue', font_color='black', font_size=8)
plt.title('Graph with Delaunay Triangulation Topology')
plt.show()


def plot_ant_colony(nodes, connections, pheromones):
    """
    Plot the ant colony graph with varying line thickness based on pheromone levels.

    Parameters:
    - nodes: List of node coordinates (e.g., [(x1, y1), (x2, y2), ...])
    - connections: Adjacency matrix indicating connections between nodes (1 if connected, 0 otherwise)
    - pheromones: Matrix representing pheromone levels on each edge

    Note: Assumes len(nodes) == len(pheromones) == len(connections)
    """
    # Separate coordinates
    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]

    # Plot connections
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if connections[i][j] == 1:
                pheromone_level = pheromones[i][j]
                color = 'black' if pheromone_level == 0 else 'orange'
                linewidth = 1 + 5 * pheromone_level  # Adjust the multiplier as needed

                plt.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=linewidth)
