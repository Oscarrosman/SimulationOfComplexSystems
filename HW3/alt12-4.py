import numpy as np
import matplotlib.pyplot as plt
import time

def InitializeAdjacencyMatrix(n, p):
    '''
    Erdős Rényi random graph
    '''
    adjMatrix = np.zeros([n, n], dtype=int)
    for i in range(n):
        for j in range(n):
            r = np.random.rand()
            if r < p and adjMatrix[i][j] == 0 and j>i:
                adjMatrix[i][j] = 1
                adjMatrix[j][i] = 1
    np.fill_diagonal(adjMatrix, 0)
    return adjMatrix

def LengthMatrix(adj):
    n = len(adj)
    matrix = -1*np.ones([n, n], dtype=int)
    comp = np.copy(matrix)
    np.fill_diagonal(comp, 0)
    t = 1
    at = np.copy(adj)

    while (np.any(comp == -1)):
        for i in range(n):
            for j in range(i, n):
                if at[i][j] != 0 and matrix[i][j] == matrix[j][i] and matrix[i][j] == -1:
                    matrix[i][j] = t
                    matrix[j][i] = t
        at = np.matmul(at, adj)
        t += 1
        comp = np.copy(matrix)
        np.fill_diagonal(comp, 0)
        if t > 100:
            print('Break')
            break

    return matrix

def AveragePathLength(n):
    Avg = lambda m, n: np.sum(m)/(n*n - n) # Elements: n*n, -n to discount diagonal
    probabilities = np.linspace(0, 1, num=100)
    avgSteps = []
    for p in probabilities:
        print(f'\nNew probability: {p}')
        at = time.time()
        adj = InitializeAdjacencyMatrix(n, p)
        lengthMatrix = LengthMatrix(adj)
        np.fill_diagonal(lengthMatrix, 0)
        avg = Avg(lengthMatrix, n)
        avgSteps.append(avg)
        print(f' > avg length: {avgSteps[-1]}, time: {(time.time() - at)} seconds')
    return avgSteps, probabilities

def CalcClusterCoefficient(adj):
    n = len(adj)
    closedTriplets = np.trace(np.linalg.matrix_power(adj, 3)) // 3
    connectedTriplets = 0
    for i in range(n):
        neighbors = np.nonzero(adj[i])[0]
        degree = len(neighbors)
        if degree >= 2:
            connectedTriplets += degree*(degree - 1) // 2
    if connectedTriplets == 0:
        return 0.0
    else:
        clusterC = closedTriplets/connectedTriplets
        return clusterC
    
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
    steps, probabilities = AveragePathLength(n)
    clusterC = ClusteringCoefficient(n, nProb)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot average steps length
    Eqn124 = lambda p, n, gamma: (1/2) + (np.log(n) - gamma) / (np.log(p*(n-1)))
    Eqn125 = lambda p: 2-p
    gamma = 0.57722
    smallP = [Eqn124(p, n, gamma) for p in probabilities]
    bigP = [Eqn125(p) for p in probabilities]
    axs[0].semilogx(probabilities[1:], steps[1:], '.')
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
    print(f'\n Simulation time: {time.time() - st:4.0f} seconds')
    plt.show()







n = 500
nP = 100
p = 0.1
st = time.time()
PlotFunction(n, nP)