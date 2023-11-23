import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def InitializeAdjecencyMatrix(n, p):
    matrix = np.zeros([n,n], dtype=int)
    for i in range(n):
        for j in range(n):
            r = np.random.rand()
            if r < p and matrix[i][j] == 0 and j>i:
                matrix[i][j] = 1
                matrix[j][i] = 1
    np.fill_diagonal(matrix, 0)
    return matrix

def InitializeNodes(n, r):
    thetas = np.linspace(0, 2*np.pi, num=n)
    nodes = [[r*np.cos(theta), r*np.sin(theta)] for theta in thetas]
    x = [r*np.cos(theta) for theta in thetas]
    y = [r*np.sin(theta) for theta in thetas]
    return nodes, x, y

def PlotFunction(nodes, adjacency, ax):
    # Seperate coordinates
    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]

    # Find connections
    pairs = []
    pairCoord = []
    for i, connections in enumerate(adjacency):
        for j, connection in enumerate(connections):
            if connection == 1:
                pairs.append([i, j])
                pairCoord.append([[x[i], x[j]], [y[i], y[j]]])
    for pair in pairCoord:
        ax.plot(pair[0], pair[1], linewidth=0.1, color='blue')
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel(f'$n = {len(nodes)}$')


    ax.scatter(x, y, color='orange')

def PlotHistogram(n, prob, adjMatrix, ax):
    # Theoretical distribution
    k = np.arange(0, n)
    pDist = binom.pmf(k, n-1, prob)
    ax.plot(k, pDist, color='orange', linewidth=2)
    ax.set_xlim([0, 30])

    # Histogram
    distribution = [sum(node) for node in adjMatrix]
    
    ax.hist(distribution, bins=np.arange(0, n + 2) - 0.5, density=True, alpha=0.7, rwidth=0.8)
    ax.set_xlabel('k')
    ax.set_ylabel('P(k)')

def MergePlots(nodes, prob, adjMatrix):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the network
    PlotFunction(nodes, adjMatrix, axs[0])
    axs[0].set_title('Network Plot')

    # Plot the histogram
    PlotHistogram(n, prob, adjMatrix, axs[1])
    axs[1].set_title('Degree Distribution')

    plt.show()

r = 1
n = 300
p = 0.05
nodes, x, y = InitializeNodes(n, r)
adj = InitializeAdjecencyMatrix(n, 0.05)
#PlotHistogram(n, p, adj)
MergePlots(nodes, p, adj)



