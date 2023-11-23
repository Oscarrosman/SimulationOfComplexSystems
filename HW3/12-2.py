import numpy as np
import matplotlib.pyplot as plt

def InitializeNodes(n, r):
    thetas = np.linspace(0, 2*np.pi, num=n)
    nodes = [[r*np.cos(theta), r*np.sin(theta)] for theta in thetas]
    x = [r*np.cos(theta) for theta in thetas]
    y = [r*np.sin(theta) for theta in thetas]
    return nodes, x, y

def InitializeAdjecencyMatrix(n, p, c):
    matrix = np.zeros([n,n], dtype=int)
    for i in range(n):
        for j in range(n):
            dist = abs(i - j)
            if j == n-1 and i == 0:
                dist = 1
            if dist >= 1 and dist <= c/2:
                matrix[i][j] = 1
                matrix[j][i] = 1
            r = np.random.rand()
            if r < p and j>i:
                matrix[i][j] = 1
                matrix[j][i] = 1
    np.fill_diagonal(matrix, 0)
    return matrix

def PlotFunction(nodes, adjacency):
    # Seperate coordinates
    x = [node[0] for node in nodes]
    print(x)
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
        plt.plot(pair[0], pair[1], linewidth=0.1, color='blue')


    plt.scatter(x, y, color='orange')
    plt.show()


n = 20
r = 3
p = 0.1
c = 4
nodes = InitializeNodes(n, r)
adj = InitializeAdjecencyMatrix(n, p, c)
print(nodes[0][0])
PlotFunction(nodes[0], adj)
