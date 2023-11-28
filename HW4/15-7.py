import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
import networkx as nx
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
    '''
    Generates or loads nodes and sets the connections between them. Then defines the distance,
    visibility and pheromone matrices.
    '''
    if mode1 == 'Create':
        nodes = np.random.rand(n, 2)
        connections = GenerateTopology(nodes)
        SaveData(nodes, 'nodes.csv')
        SaveData(connections, 'ConnectionMatrix.csv')
    else:
        nodes = LoadData('nodes.csv')
        connections = LoadData('ConnectionMatrix.csv')
    
    distance, weights = GenerateDistanceVisibility(nodes, connections)
    pheromoneLevels = np.copy(connections)
    return nodes, connections, pheromoneLevels, distance, weights

def GenerateDistanceVisibility(nodes, connections):
    n = len(nodes)
    distanceMatrix = np.zeros([n, n], dtype=float)
    weightMatrix = np.zeros([n, n], dtype=float)
    Euclidian = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    for i, node in enumerate(connections):
        for j, path in enumerate(node):
            if path == 1:
                distanceMatrix[i][j] = distanceMatrix[j][i] = Euclidian(nodes[i], nodes[j])
                weightMatrix[i][j] = weightMatrix[j][i] = 1/Euclidian(nodes[i], nodes[j])
    return distanceMatrix, weightMatrix

def GenerateTopology(points):
    n = len(points)
    connectionMatrix = np.zeros([n, n], dtype=int)
    if mode2 == 'D':
        triangulation  = sp.Delaunay(points)
        connections = []
        for simplex in triangulation.simplices:
            connections.extend([(simplex[i], simplex[(i + 1) % 3]) for i in range(3)])
        for connection in connections:
            i, j = connection
            connectionMatrix[i][j] = 1
            connectionMatrix[j][i] = 1
    np.fill_diagonal(connectionMatrix, 0)
    return connectionMatrix

def GeneratePath(start, end, connections, visibility, pheromoneLevels, distances):
    currentNode = start
    path = [start]
    nodes = np.linspace(0, len(connections)-1, num=len(connections), dtype=int)
    distance = []
    while currentNode != end:
        p = []
        for i in range(len(pheromoneLevels[currentNode])):
            probability = (pheromoneLevels[currentNode][i]**alpha)*(visibility[currentNode][i]**beta)
            p.append(probability)
        probabilities = [pNode/np.sum(p) for pNode in p]
        nextNode = np.random.choice(nodes, p=probabilities)
        path.append(nextNode)
        #distance += distances[currentNode][nextNode]
        #print(distance, distances[currentNode][nextNode])
        distance.append(distances[currentNode][nextNode])
        currentNode = nextNode
    print(f'Distance calculation {np.sum(distance)}')
    return path

def PathLength(path, nodes, distances):
    dist = 0
    for i in range(len(path)):
        if i == 0:
            continue
        else:
            dist += distances[path[i]-1][path[i]]
    print('In function: ', dist)


def RunAlgortihm(n, nAnts, s, start='random', end='random'):

    # Initialize grid
    nodes, connections, pheromoneLevels, distannces, visibility = Initialization(n)

    if start == 'random':
        startPoint = np.random.randint(0, n-1)
    else:
        startPoint = start
    if end == 'random':
        endPoint = np.random.randint(0, n-1)
        while endPoint == startPoint:
            endPoint = np.random.randint(0, n-1)
    else:
        endPoint = end



    for step in range(s):
        antPaths = []
        pathLengths = np.zeros([1, nAnts])
        for i in range(nAnts):
            pass
        pass
    


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
        counter += 1
        plt.plot(pair[0], pair[1], linewidth=0.1, color='blue')

    plt.xticks([]), plt.yticks([])
    plt.scatter(x, y, color='orange')


    plt.show()




mode1 = 'Create'
mode2 = 'D' # D = Delaunay
#mode = 'Load'
alpha = 0.8
beta = 1
decay = 0.5
nodes, connections, pheromone, dist, eta = Initialization(10)
#PlotFunction(nodes, connections)
path = GeneratePath(6, 1, connections, eta, pheromone, dist)
print(path)
PathLength(path, nodes, dist)
#GenerateTopology(a)