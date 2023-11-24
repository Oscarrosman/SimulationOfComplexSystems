import numpy as np
import matplotlib.pyplot as plt

def InitializeAdjacencyMatrix(n, n0, m):
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

def LengthMatrix(adj):
    # Initialize Matrix
    n = len(adj)
    
    matrix = -1*np.ones([n, n], dtype=int)
    np.fill_diagonal(matrix, 0)
    at = adj
    t = 0
    while np.any(matrix == -1):
        t += 1
        at = np.linalg.matrix_power(adj, t)
        for i in range(n):
            for j in range(n):
                if at[i][j] != 0 and matrix[i][j] == -1:
                    matrix[i][j] = t
                    matrix[j][i] = t
        if t > 10:
            print('Probability cancelled, t > 10')
            matrix = 10*np.ones([n, n], dtype=int)
            break

    np.fill_diagonal(matrix, -1)
    return matrix

def AveragePathLength(n, nProb):
    probabilities = np.linspace(0, 1, num=nProb)
    avgSteps = []
    for p in probabilities:
        adj = InitializeAdjacencyMatrix(n, p)
        lenMatrix = LengthMatrix(adj)
        avgSteps.append(np.mean(lenMatrix))
    return avgSteps, probabilities

def AltCC(adj):
    a3 = np.linalg.matrix_power(adj, 3)
    closedTriplets = np.sum(a3)
    k = np.sum(adj, 1)
    ki = [ki*(ki - 1) for ki in k]
    allTriplets = np.sum(ki)
    if allTriplets == 0:
        return 0.0
    else:
        clusterC = closedTriplets/allTriplets
        clusterC = clusterC/500
        return clusterC

def ClusteringCoefficient(n, nProb):
    probabilities = np.linspace(0, 1, num=nProb)
    clusterC = []
    for p in probabilities:
        adj = InitializeAdjacencyMatrix(n, p)
        clusterC.append(AltCC(adj))
    return clusterC

def PlotFunction(n, nProb):
    steps, probabilities = AveragePathLength(n, nProb)
    clusterC = ClusteringCoefficient(n, nProb)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot average steps length
    Eqn124 = lambda p, n, gamma: (1/2) + (np.log(n) - gamma) / (np.log(p*(n-1)))
    Eqn125 = lambda p: 2-p
    gamma = 0.57722
    smallP = [Eqn124(p, n, gamma) for p in probabilities]
    bigP = [Eqn125(p) for p in probabilities]
    axs[0].loglog(probabilities[1:], steps[1:], '.')
    axs[0].plot(probabilities[1:], smallP[1:], color='green', linewidth=1)
    axs[0].plot(probabilities[1:], bigP[1:], color='red', linewidth=1)
    axs[0].set_xlim([10**(-2), 1.1*10**(0)])
    axs[0].set_ylim([0.5, 4])
    axs[0].set_xlabel('p')
    axs[0].set_ylabel('length')

    # Plot Clustering coefficient
    axs[1].plot(probabilities, clusterC, '.')
    axs[1].plot(probabilities, probabilities, color='red', linewidth=1)
    axs[1].set_xlabel('p')
    axs[1].set_ylabel('C')
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])

    plt.show()

    n = 1000
    m = [1,3,10]