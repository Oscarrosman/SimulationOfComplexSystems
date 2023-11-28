import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
import networkx as nx
from collections import Counter
import csv, time, os

def SaveData(data, filename):
    folder = 'Data'
    filePath = os.path.join(os.getcwd(), folder, filename)
    with open(filePath, 'w', newline='') as file:
        writer=csv.writer(file)
        writer.writerows(data)

def LoadData(filename):
    data = []
    folder = 'Data'
    filePath = os.path.join(os.getcwd(), folder, filename)
    with open(filePath, 'r') as file:
        for line in file:
            line = line.split(',')
            line = [float(i) for i in line]
            data.append(line)
    return data

def Initialization(n):
    if mode1 == 'Create':
        nodes = np.random.rand(n, 2)
        connections = GenerateTopology(nodes)
        # Save data
        SaveData(nodes, 'nodes.csv')
        SaveData(connections, 'ConnectionMatrix.csv')
    else:
        # Load data
        nodes = LoadData('nodes.csv')
        connections = LoadData('ConnectionMatrix.csv')
    
    return nodes, connections

def GenerateTopology(points):
    n = len(points)
    # Define matrix
    connectionMatrix = np.zeros([n, n], dtype=int)
    # Find connections
    triangulation  = sp.Delaunay(points)
    # Reformat connections
    connections = []
    for simplex in triangulation.simplices:
        connections.extend([(simplex[i], simplex[(i + 1) % 3]) for i in range(3)])
    for connection in connections:
        i, j = connection
        connectionMatrix[i][j] = 1
        connectionMatrix[j][i] = 1
    # Ensure the diagonal is filled with zeros
    np.fill_diagonal(connectionMatrix, 0)
    return connectionMatrix

def FindDistances(nodes, connections):
    '''
    Calculates the weights and distances between all nodes if there is a connection between them.
    '''
    Euclidian = lambda p1, p2: np.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))
    distanceMatrix = np.zeros([len(nodes), len(nodes)], dtype=float)
    weights = np.copy(distanceMatrix)
    for i, nodeConnections in enumerate(connections):
        for j, path in enumerate(nodeConnections):
            # Distance: Euclidian distance
            distanceMatrix[i][j] = distanceMatrix[j][i] = Euclidian(nodes[i], nodes[j])
            # Weight: 1/dij
            weights[i][j] = weights[j][i] = 1/Euclidian(nodes[i], nodes[j])
    # Ensure diagonal is filled with zeros
    np.fill_diagonal(distanceMatrix, 0)
    np.fill_diagonal(weights, 0)
    return distanceMatrix, weights

def TraveledDistance(path, distanceMatrix):
    distance = 0
    for i, step in enumerate(path):
        if i < len(path)-1:
            distance += distanceMatrix[step][path[i+1]]
    return distance

def TakeStep(nodes, connections, currentNode, weights, pheromonelevels):
    probabilities = []
    for i in range(len(connections[currentNode])):
        probabilities.append((pheromonelevels[currentNode][i]**alpha)*(weights[currentNode][i]**beta))

    probabilities = [p/np.sum(probabilities) for p in probabilities]
    nextNode = np.random.choice(nodes, p=probabilities)
    return nextNode

def SimplifyPath(path):
    '''
    Can be improved!!
    '''
    visitedForward = [path[0]]
    visitedBackward = []
    for i in range(1, len(path)):
        visitedForward.append(path[i])
        visitedBackward.append(path[-i])
        if visitedBackward[-1] in visitedForward:
            idx = visitedForward.index(visitedBackward[-1])
            visitedForward = visitedForward[:idx]
            break
        elif visitedForward[-1] in visitedBackward:
            idx = visitedBackward.index(visitedForward[-1])
            visitedBackward = visitedBackward[:idx]
            break
    
    visitedBackward.reverse()
    simplifiedPath = visitedForward + visitedBackward
    return simplifiedPath

def RunAlgorithm(n, nAnts, stop, start = -1, end=-1):
    # Initialize grid
    nodes, connections = Initialization(n)
    pheromoneLevels = np.copy(connections)
    nodeOptions = np.linspace(0, n-1, num=n, dtype=int)
    dMatrix, weights = FindDistances(nodes, connections)
    if start == -1:
        startPoint = np.random.randint(0, n-1)
    else:
        startPoint = start
    if end == -1:
        endPoint = np.random.randint(0, n-1)
    else:
        endPoint = end


    # Initialize ants
    antPaths = [[startPoint] for i in range(nAnts)]
    travelDistance = [0 for i in range(nAnts)]

    # Generate paths
    for step in range(stop):
        for ant in range(nAnts):
            if antPaths[ant][-1] != endPoint:
                antPaths[ant].append(TakeStep(nodeOptions, connections, antPaths[ant][-1], weights, pheromoneLevels))
    
    # Calculate distance of finished paths
    for ant in range(nAnts):
        if antPaths[ant][-1] == endPoint:
            antPaths[ant] = SimplifyPath(antPaths[ant])
            travelDistance[ant] = TraveledDistance(antPaths[ant], dMatrix)

    PlotPaths(antPaths, nodes, connections, startPoint, endPoint)
    # Simply path before distance calculation, done but can be improved
    # Update pheromone levels

def PlotPath(paths, nodes, connections, start, end):
    # Check how many paths reached the end:
    counter = 0
    completedPaths = []
    for i,path in enumerate(paths):
        if path[-1] == end:
            counter += 1
            completedPaths.append(i)

    figure, axs = plt.subplots(1, counter)
    for i in range(len(completedPaths)):
        pass


def PlotPaths(paths, nodes, connections, start, end):
    # Check how many paths reached the end:
    counter = 0
    completedPaths = []
    for i,path in enumerate(paths):
        if path[-1] == end:
            counter += 1
            completedPaths.append(i)

    figure, axs = plt.subplots(1, counter)
    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]

    for k, ax in enumerate(axs):
        # Find and plot ALL connections
        pairs = []
        pairCoord = []
        for i, connections in enumerate(connections):
            for j, connection in enumerate(connections):
                if connection == 1:
                    pairs.append([i, j])
                    pairCoord.append([[x[i], x[j]], [y[i], y[j]]])
        for pair in pairCoord:
            ax.plot(pair[0], pair[1], linewidth=0.1, color='blue')
    
        # Plot path:
        print(completedPaths)
        print(len(paths), k)
        path = paths[completedPaths[k]]
        print(path)
        xPath = [x[edge] for edge in path]
        yPath = [y[edge] for edge in path]
        ax.plot(xPath, yPath, linewidth=1, color='red')

        # Add nodes
        ax.scatter(x, y, color='black')
        # Highlight start and end nodes
        ax.scatter([x[start], x[end]], [y[start], y[end]], color='green')
        plt.show()
    plt.show()






# Variables
mode1 = 'Load' # 'Load
alpha = 0.8
beta = 1

RunAlgorithm(40, 5, 100)
