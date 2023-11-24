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
            for neighbor in range(c):
                matrix[i][(i+int(neighbor/2)+1)%n] = 1
                matrix[(i+int(neighbor/2)+1)%n][i] = 1
            r = np.random.rand()
            if r < p and j>i:
                matrix[i][j] = 1
                matrix[j][i] = 1
    np.fill_diagonal(matrix, 0)
    return matrix

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

def AveragePathLength(n, c, nProb):
    Avg = lambda m, n: np.sum(m)/(n*n - n) # Elements: n*n, -n to discount diagonal
    probabilities = np.logspace(-5, 0, 35)
    avgSteps = []
    for p in probabilities:
        print(f'\nNew probability: {p}')
        at = time.time()
        adj = InitializeAdjacencyMatrix(n, p, c)
        lengthMatrix = LengthMatrix(adj)
        np.fill_diagonal(lengthMatrix, 0)
        avg = Avg(lengthMatrix, n)
        avgSteps.append(avg)
        print(f' > avg length: {avgSteps[-1]}, time: {(time.time() - at)} seconds')
    return avgSteps, probabilities

def ClusterCoeffiecient(adj):
    a2 = np.linalg.matrix_power(adj, 2)
    a3 = np.linalg.matrix_power(adj, 3)
    d = np.diagonal(a3)
    d = np.sum(d)
    degrees = [np.sum(node) for node in adj]
    degrees = [node**2 - node for node in degrees]
    coefficient = d / np.sum(degrees)
    return coefficient

def GetClusterCoefficients():
    '''
    Calculates the clustering coefficients for n = 100 and n = 1000
    '''
    TheoreticalCC = lambda c: (3*(c-2)/(4*(c-1)))
    coeffiecients = np.arange(start=2, stop=100, step=2).tolist() + np.arange(start=100, stop=1100, step=100).tolist()
    realC100 = []
    realC1000 = []
    theoreticalC = []
    for i, c in enumerate(coeffiecients):
        print(f'\nNew coefficient: {c}')
        at = time.time()
        adj100 = InitializeAdjacencyMatrix(100, 0, c)
        adj1000 = InitializeAdjacencyMatrix(1000, 0, c)
        c100 = ClusterCoeffiecient(adj100)
        c1000 = ClusterCoeffiecient(adj1000)
        realC100.append(c100)
        realC1000.append(c1000)
        theoreticalC.append(TheoreticalCC(c))
        print(f' > time: {(time.time() - at)} seconds\n > Process: {i/len(coeffiecients)}')
    return realC100, realC1000, theoreticalC, coeffiecients

def PlotFunction(n, c, nProb):
    steps, probabilities = AveragePathLength(n, c, nProb)
    c100, c1000, ct, coeff= GetClusterCoefficients()

    fig, axs = plt.subplots(1, 2, figsize=(10,5))

    # Plot average steps length
    Eqn124 = lambda p, n, gamma: (1/2) + (np.log(n) - gamma) / (np.log(p*(n-1)))
    Eqn125 = lambda p: 2-p
    gamma = 0.57722
    smallP = [Eqn124(p, n, gamma) for p in probabilities]
    bigP = [Eqn125(p) for p in probabilities]
    axs[0].semilogx(probabilities, steps, '.')
    axs[0].axhline(n/(2*c), label="n/2c", color="red")
    axs[0].axhline(np.log(n)/np.log(c), label="ln(n)/ln(c)", color="green")
    axs[0].set_xlim(10e-6, 10e0)
    axs[0].set_ylim([0, 50])
    axs[0].set_xlabel('p')
    axs[0].set_ylabel('length')

    # Plot coefficients
    axs[1].semilogx(coeff, c100, '.', color='blue')
    axs[1].semilogx(coeff, c1000, '.', color='orange')
    axs[1].semilogx(coeff, ct, '--', color='black')
    axs[1].set_ylim(0,1)
    print(f'Simulation time: {(time.time() - st)%60} minutes')
    plt.show()

n = 500
p = 0
c = 6
nP = 100
st = time.time()
PlotFunction(n, c, nP)

