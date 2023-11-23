import numpy as np
import matplotlib.pyplot as plt
import time

def InitializeAdjacencyMatrix(n, p, c):
    '''
    Watts Strogatz small-world graph
    '''
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
        if t > 100:
            print('Probability cancelled, t > 10')
            matrix = 10*np.ones([n, n], dtype=int)
            break

    np.fill_diagonal(matrix, 0)
    return matrix

def AveragePathLength(n, nProb):
    probabilities = np.linspace(10**(-5), 10**(-3), num=int(nProb/2)).tolist()
    probabilities.append(np.linspace(10**(-2), 1, num=int(nProb/2)))
    probabilities = [np.linspace(10**(-5), 10**(-3), num=int(nProb/2)).tolist(), np.linspace(10**(-2), 1, num=int(nProb/2)).tolist()]
    avgSteps = []
    for prob in probabilities:
        for p in prob:
            print(f'\nNew probability: {p}')
            at = time.time()
            adj = InitializeAdjacencyMatrix(n, p, c)
            lenMatrix = LengthMatrix(adj)
            avgSteps.append(np.mean(lenMatrix))
            print(f' > avg length: {avgSteps[-1]}, time: {(time.time() - at)} seconds')
    return avgSteps, probabilities

def PlotFunction(n, nProb):
    steps, p = AveragePathLength(n, nProb)
    probabilities = []
    for i in p:
        for j in i:
            probabilities.append(j)
    #clusterC = ClusteringCoefficient(n, nProb)

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    # Plot average steps length
    Eqn126 = lambda p, n, gamma: n/(2*c)
    Eqn127 = lambda p: np.log(n)/np.log(c)
    gamma = 0.57722
    smallP = [Eqn126(p, n, gamma) for p in probabilities]
    bigP = [Eqn127(p) for p in probabilities]
    axs.loglog(probabilities, steps, '.')
    axs.plot(probabilities, smallP, color='green', linewidth=1)
    axs.plot(probabilities, bigP, color='red', linewidth=1)
    axs.set_xlim([10**(-5), 1.1])
    axs.set_ylim([0, 50])
    axs.set_xlabel('p')
    axs.set_ylabel('length')
    print(f'Simulation time: {(time.time() - st)%60} minutes')
    plt.show()

n = 500
c = 6
p = 0
nP = 100
st = time.time()
probabilities = np.linspace(10**(-5), 10**(-3), num=int(nP/2)).tolist()
probabilities.append(np.linspace(10**(-2), 1, num=int(nP/2)).tolist())
probabilities = [np.linspace(10**(-5), 10**(-3), num=int(nP/2)).tolist(), np.linspace(10**(-2), 1, num=int(nP/2)).tolist()]
PlotFunction(n, nP)
