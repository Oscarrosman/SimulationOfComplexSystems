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
    if mode == 'Create':
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

def DistanceMatrix(nodes, connections):
    '''
    Function to calculate distance and weight matrix
    '''
    n = len(nodes)
    # Define distance metrix and matrix
    Euclidian = lambda p1, p2: np.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))
    dMatrix = np.zeros([n, n], dtype=float)
    weights = np.zeros([n, n], dtype=float)

    for i in range(n):
        for j in range(n):
            if connections[i][j] == 1:
                # Distance: Euclidian
                dMatrix[i][j] = dMatrix[j][i] = Euclidian(nodes[i], nodes[j])
                # Weight: 1/dij
                weights[i][j] = weights[j][i] = 1 / Euclidian(nodes[i], nodes[j])
    np.fill_diagonal(dMatrix, 0)
    np.fill_diagonal(weights, 0)
    return dMatrix, weights

def PathLength(path, dMatrix):
    distance = 0
    for i, currentNode in enumerate(path):
        if currentNode == path[-1]:
            break
        nextNode = path[i+1]
        distance += dMatrix[currentNode][nextNode]
    return distance

def TakeStep(nodes, connections, currentNode, weights, pheromoneLevels):
    probabilities = []
    n = len(connections)
    SingleProbability = lambda tao, w: (tao**alpha)
    for i in range(n):
        probabilities.append(SingleProbability(pheromoneLevels[currentNode][i],weights[currentNode][i]))
    
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

def UpdatePheromone(pheromoneLevels, paths, pDistance, end, connections):
    # Apply decay to all pheromone levels (1-p)*tao
    for i in range(len(pheromoneLevels)):
        for j in range(len(pheromoneLevels[0])):
            if connections[i][j] == 1:
                pheromoneLevels[i][j] *= (1-decay)
                if pheromoneLevels[i][j] < (1/(10**5)):
                    pheromoneLevels[i][j] = (1/(10**5))
    
    # Calculate added pheromone
    for i,path in enumerate(paths):
        edges = []
        if path[-1] == end:
            delta = Q/pDistance[i]
            for j in range(len(path)-1):
                edges.append([path[j], path[j+1]])
            for j in range(len(edges)):
                pheromoneLevels[edges[j][0]][edges[j][1]] += delta
            
    return pheromoneLevels

def PlotLength(r, sPathLength, mPathLength):
    rs = np.linspace(0, r, num=len(sPathLength))
    plt.subplot(1, 2, 1)
    plt.plot(rs, sPathLength, color='blue')
    plt.title('Shortest path length')
    plt.xlabel('Rounds')
    plt.ylabel('Length')
    plt.subplot(1, 2, 2)
    plt.title('Mean path length')
    plt.plot(rs, mPathLength, color='red')
    plt.xlabel('Rounds')
    plt.ylabel('Length')
    plt.show()

def PlotPath(nodes, path, start, end, step, dMatrix):
    print(f'Shortest path round {step}: {PathLength(path, dMatrix)}')
    ## Plot path ##
    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]
    xPath = [x[edge] for edge in path]
    yPath = [y[edge] for edge in path]
    plt.plot(xPath, yPath, linewidth=2, color='red')
    ## Plot all nodes and connections ##
    # Perform Delaunay triangulation
    triangulation = sp.Delaunay(nodes)
    # Create a graph from the Delaunay triangulation
    graph = nx.Graph()
    # Add nodes and edges to the graph
    for simplex in triangulation.simplices:
        graph.add_edge(simplex[0], simplex[1])
        graph.add_edge(simplex[1], simplex[2])
        graph.add_edge(simplex[2], simplex[0])
    # Draw the graph
    pos = {i: nodes[i] for i in range(len(nodes))}
    nx.draw_networkx(graph, pos, with_labels=True, font_weight='bold', node_size=200, node_color='skyblue', font_color='black', font_size=8)
    plt.title(f'Shortest path on round: {step}')

    plt.show()

def PlotPheromone(nodes, connections, pMatrix, step):
    n = len(nodes)
    # Modify connection lines based on pheromone levels
    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]

    for i in range(n):
        for j in range(n):
            if connections[i][j] == 1:
                pLevel = pMatrix[i][j] + pMatrix[j][i]
                color = 'black' if pLevel < (1/(10**5)) else 'orange'
                if step == 0:
                    linewidth = 2.5*pLevel
                else:
                    linewidth = 0.2*pLevel
                plt.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=linewidth)

    ## Plot all nodes and connections ##
    # Perform Delaunay triangulation
    triangulation = sp.Delaunay(nodes)
    # Create a graph from the Delaunay triangulation
    graph = nx.Graph()
    # Add nodes and edges to the graph
    for simplex in triangulation.simplices:
        graph.add_edge(simplex[0], simplex[1])
        graph.add_edge(simplex[1], simplex[2])
        graph.add_edge(simplex[2], simplex[0])
    # Draw the graph
    pos = {i: nodes[i] for i in range(n)}
    nx.draw_networkx(graph, pos, with_labels=True, font_weight='bold', node_size=200, node_color='skyblue', font_color='black', font_size=8)
    plt.title(f'Pheromone levels on round: {step}')
    plt.show()

def RunAlgorithm(r, s, nAnts, n, savePoints):
    # Initialize grid and matrices
    nodes, connections = Initialization(n)
    dMatrix, weights = DistanceMatrix(nodes, connections)
    pMatrix = np.copy(connections) # Pheromone levels initialized to 1 where connections exist
    nodeList = np.linspace(0, n-1, num=n, dtype=int)

    # Set start and stop points:
    start = np.random.randint(0, n-1)
    end = np.random.randint(0, n-1)
    start = 8
    end = 31
    while start == end:
        end = np.random.randint(0, n-1)
    print(f'\n Start point: {start}\n End point: {end}')

    # Set up data save for later plotting
    sPath = []
    sPathLength = []
    mPathLength = []

    for round in range(r):
        # Generate ant paths, all at the same start point
        antPaths = [[start] for i in range(nAnts)]
        for step in range(s):
            for ant in range(nAnts):
                if antPaths[ant][-1] != end: # Only take step if end is not reached
                    antPaths[ant].append(TakeStep(nodeList, connections, antPaths[ant][-1], weights, pMatrix))
        
        # Simplify paths and calculate length of finished paths
        pLength = []
        fPaths = []
        for ant in range(nAnts):
            if antPaths[ant][-1] == end:
                fPaths.append(SimplifyPath(antPaths[ant])) # Save finished paths seperately
                pLength.append(PathLength(antPaths[ant], dMatrix))
        
        # Update pheromone levels, only finished paths influence
        pMatrix = UpdatePheromone(pMatrix, fPaths, pLength, end, connections)

        if round < 20:
            idx = np.argmin(pLength)
            sPathLength.append(np.copy(pLength[idx]))
            mPathLength.append(np.mean(pLength))
        elif round % 10 == 0 and round > 20:
            idx = np.argmin(pLength)
            sPathLength.append(np.copy(pLength[idx]))
            mPathLength.append(np.mean(pLength))

        if round in savePoints:
            idx = np.argmin(pLength)
            PlotPheromone(nodes, connections, pMatrix, round)
            PlotPath(nodes, fPaths[idx], start, end, round, dMatrix)
            sPath.append(fPaths[idx])
    #PlotPheromone(nodes, connections, pMatrix)
    #PlotPath(nodes, connections, fPaths[-1], start, end)
    PlotLength(r, sPathLength, mPathLength)
        

# Variables
mode = 'Load'
alpha = 0.8
beta = 1
decay = 0.5
R = 300
S = 80
nA = 20
Q = 1


RunAlgorithm(R, S, nA, 40, [0, 15, 50])
