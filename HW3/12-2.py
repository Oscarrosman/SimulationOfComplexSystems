import numpy as np
import matplotlib.pyplot as plt

def InitializeNodes(n, r):
    
    thetas = np.linspace(0, 2*np.pi, num=n)
    nodes = [[r*np.cos(theta), r*np.sin(theta)] for theta in thetas]
    #x = [r*np.cos(theta) for theta in thetas]
    #y = [r*np.sin(theta) for theta in thetas]
    return nodes#, x, y

def InitializeAdjacencyMatrix(n, p, c):
    '''
    Watts Strogatz small-world graph
    '''
    matrix = np.zeros([n,n], dtype=int)
    for i in range(n):
        for j in range(n):
            for neighbor in range(c):
                matrix[i][(i+int(neighbor/2)+1)%n] = 1
                matrix[(i+int(neighbor/2)+1)%n][i] = 1
            r = np.random.rand()
            if r < p and j>i:
                matrix[i][j] = 1
                matrix[j][i] = 1
    np.fill_diagonal(matrix, 0)
    print(matrix)
    return matrix

def PlotFunction(nodes, adjacency):
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

    counter = 1
    for pair in pairCoord:
        print(counter)
        counter += 1
        plt.plot(pair[0], pair[1], linewidth=0.1, color='blue')

    plt.xticks([]), plt.yticks([])
    plt.scatter(x, y, color='orange')
    plt.show()


n = 20
r = 1
p = 0
c = 4
nodes = InitializeNodes(n, r)
adj = InitializeAdjacencyMatrix(n, p, c)
PlotFunction(nodes, adj)
