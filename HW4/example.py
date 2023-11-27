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