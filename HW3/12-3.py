import numpy as np
import matplotlib.pyplot as plt

def InitializeAdjacencyMatrix(n, n0, m):
    '''
    The Albert Barabási preferential-growth model
    '''
    matrix = np.ones([n0, n0], dtype=int)
    np.fill_diagonal(matrix, 0)
    matrix = matrix.tolist()

    for i in range(n-n0):
        # Determine connections to new node
        nodes, connections = [i for i, node in enumerate(matrix)],[sum(node) for node in matrix]
        probabilities = [p/sum(connections) for p in connections]
        c = np.random.choice(nodes, size=m, p=probabilities)
        # Create new node
        temp = np.zeros(len(matrix), dtype=int).tolist()
        matrix.append(temp)
        for i in range(len(matrix)):
            matrix[i].append(0)
        for connection in c:
            matrix[connection][-1] = 1
            matrix[-1][connection] = 1
    matrix = np.array(matrix)
    return matrix

def GenerateNodes(adjacency, r):
    thetas = np.linspace(0, 2*np.pi, num=len(adjacency))
    nodes = [[r*np.cos(theta), r*np.sin(theta)] for theta in thetas]
    return nodes

def FindDegrees(adjacency):
    degrees = np.sum(adjacency, 1)
    u = [i/len(adjacency) for i,degree in enumerate(degrees)]
    degrees = sorted(degrees, reverse=True)
    return degrees, u

def PlotFunction(nodes, adjacency):
    # Define subplots
    fig, axs = plt.subplots(1, 2, figsize=(10,5))

    # Plot network
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
        axs[0].plot(pair[0], pair[1], linewidth=0.1, color='blue')
    axs[0].scatter(x, y, color='orange')
    axs[0].set_title('Network model')
    axs[0].set_xlabel(f'n = {len(adjacency)}')
    axs[0].set_xticks([]), axs[0].set_yticks([])

    # Plot degree distribution
    degrees, u = FindDegrees(adjacency)
    axs[1].loglog(degrees, u, '.', color='orange')
    ks = np.linspace(m, max(degrees))
    ck = [(m**2)/(k**(2)) for k in ks]
    axs[1].loglog(ks, ck, '--', color='grey')
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('C(k)')
    axs[1].set_title('Degree distribution')
    plt.show()


n = 100
m = 3
n0 = 5
r = 1

adj = InitializeAdjacencyMatrix(n, n0, m)
nodes = GenerateNodes(adj, r)
PlotFunction(nodes, adj)
