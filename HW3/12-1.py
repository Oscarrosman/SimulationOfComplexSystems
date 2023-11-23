import numpy as np
import matplotlib.pyplot as plt

def InitializeAdjecencyMatrix(n, p):
    matrix = np.zeros([n,n], dtype=int)
    for i in range(n):
        for j in range(n):
            r = np.random.rand()
            if r < p and matrix[i][j] == 0 and i !=j:
                matrix[i][j] = 1
                matrix[j][i] = 1
    return matrix

def InitializeNodes(n, r):
    thetas = np.linspace(0, 2*np.pi, num=n)
    nodes = [[r*np.cos(theta), r*np.sin(theta)] for theta in thetas]
    x = [r*np.cos(theta) for theta in thetas]
    y = [r*np.sin(theta) for theta in thetas]
    return nodes, x, y

def PlotFunction(nodes, adjacency, r):
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
    print(len(pairCoord))
    for pair in pairCoord:
        plt.plot(pair[0], pair[1], linewidth=0.1, color='blue')


    plt.scatter(x, y, color='orange')


    plt.show()

def PlotHistogram(n, p, samples):
    degrees = []
    for _ in range(samples):
        degree_seq = InitializeAdjecencyMatrix(n, p)
        degrees.extend(degree_seq)
    
    plt.hist(degrees, bins=np.arange(0, n + 2) - 0.5, density=True, alpha=0.7)
    plt.xlabel('Degree (k)')
    plt.ylabel('Probability Density')
    plt.title(f'Degree Histogram for Erdős–Rényi Random Graph (n={n}, p={p})')
    plt.show()
    
r = 3
n = 100
nodes, x, y = InitializeNodes(n, r)
ajd = InitializeAdjecencyMatrix(n, 0.05)
print(ajd)
#PlotFunction(nodes, ajd, r)

n_value = 20
p_value_example = 0.2
num_samples_example = 10
PlotHistogram(n_value, p_value_example, num_samples_example)
help(list.extend)