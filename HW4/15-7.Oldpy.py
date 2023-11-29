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
    connectionMatrix = np.zeros([n, n], dtype=float)
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
    print(distanceMatrix)
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

def UpdatePheromone(pheromoneLevels, paths, travelDistance, end):
    # Apply decay to all pheromone levels (1-p)*tao
    for i in range(len(pheromoneLevels)):
        for j in range(len(pheromoneLevels[0])):
            pheromoneLevels[i][j] *= (1-decay)
    # Calculate added pheromone
    for i,path in enumerate(paths):
        edges = []
        if path[-1] == end:
            delta = 1/travelDistance[i]
            for j in range(len(path)-1):
                edges.append([path[j], path[j+1]])
            for j in range(len(edges)):
                pheromoneLevels[edges[j][0]][edges[j][1]] += delta
            
    return pheromoneLevels

def RunAlgorithm(nAnts, stop, start, end, nodeOptions, connections, weights, distanceMatrix, pheromoneLevels):
    # Initialize ants
    antPaths = [[start] for i in range(nAnts)]
    travelDistance = [10**3 for i in range(nAnts)]

    # Generate paths
    for step in range(stop):
        for ant in range(nAnts):
            if antPaths[ant][-1] != end:
                antPaths[ant].append(TakeStep(nodeOptions, connections, antPaths[ant][-1], weights, pheromoneLevels))
    
    # Calculate distance of finished paths
    for ant in range(nAnts):
        if antPaths[ant][-1] == end:
            antPaths[ant] = SimplifyPath(antPaths[ant])
            travelDistance[ant] = TraveledDistance(antPaths[ant], distanceMatrix)

    # Update pheromone levels
    pheromoneLevels = UpdatePheromone(pheromoneLevels, antPaths, travelDistance, end)

    return pheromoneLevels, antPaths, travelDistance

def PlotPath(path, connections, nodes, start, end):
    # Seperate coordinates:
    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]
    plt.plot([x[start], x[end]], [y[start], y[end]], '--',color='green')
    # Plot connections
    pairs = []
    pairCoord = []
    for i, connections in enumerate(connections):
        for j, connection in enumerate(connections):
            if connection == 1:
                pairs.append([i, j])
                pairCoord.append([[x[i], x[j]], [y[i], y[j]]])
    for pair in pairCoord:
        plt.plot(pair[0], pair[1], linewidth=0.1, color='black')

    # Plot path
    xPath = [x[edge] for edge in path]
    yPath = [y[edge] for edge in path]
    plt.plot(xPath, yPath, linewidth=1, color='red')

    # Plot nodes
    plt.scatter(x, y, color='black')
    plt.scatter([x[start], x[end]], [y[start], y[end]], color='green')
    plt.title('Shorest path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def Main(rounds, n, nAnts, stop, start=-1, end=-1):
    # Initialize grid
    nodes, connections = Initialization(n)
    pheromoneLevels = np.copy(connections)
    nodeOptions = np.linspace(0, n-1, num=n, dtype=int)
    dMatrix, weights = FindDistances(nodes, connections)

    # Set start and end points:
    if start == -1:
        startPoint = np.random.randint(0, n-1)
    else:
        startPoint = start
    if end == -1:
        endPoint = np.random.randint(0, n-1)
        # Ensure they are not the same point
        while endPoint == startPoint:
            endPoint = np.random.randint(0, n-1)
    else:
        endPoint = end

    # Data save:
    sPath = []
    for i in range(rounds):
        pheromoneLevels, paths, tDistance = RunAlgorithm(nAnts, stop, startPoint, endPoint, nodeOptions, connections, weights, dMatrix, pheromoneLevels)
        sPath.append(np.min(tDistance))
    
    print(sPath)
    r = np.linspace(0, rounds, num=rounds)
    plt.plot(r, sPath)
    plt.show()
        
# Variables
mode1 = 'Load' # 'Load
alpha = 0.8
beta = 1
decay = 0.5

Main(100, 10, 5, 100)