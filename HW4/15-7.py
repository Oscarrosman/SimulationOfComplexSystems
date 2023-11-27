import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
import networkx as nx
import csv, time

def SaveData(data, filename):
    with open(filename, 'w', newline='') as file:
        writer=csv.writer(file)
        writer.writerows(data)

def LoadData(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.split(',')
            line = [float(i) for i in line]
            data.append(line)
    return data

def InitializeGraph(n):
    if mode1 == 'Create':
        nodes = np.random.rand(n, 2)
        connections = GenerateTopology(nodes)
        SaveData(nodes, 'nodes.csv')
        SaveData(connections, 'ConnectionMatrix.csv')
    else:
        nodes = LoadData('nodes.csv')
        connections = LoadData('ConnectionMatrix.csv')
    return nodes, connections

def GenerateTopology(nodes):
    connectionMatrix = np.zeros([len(nodes), len(nodes)], dtype=int)
    if mode2 == 'D':
        triangulation  = sp.Delaunay(nodes)
        connections = []
        for simplex in triangulation.simplices:
            connections.extend([(simplex[i], simplex[(i + 1) % 3]) for i in range(3)])
        for connection in connections:
            i, j = connection
            connectionMatrix[i][j] = 1
            connectionMatrix[j][i] = 1
    return connectionMatrix



mode1 = 'Create'
mode2 = 'D' # D = Delaunay
#mode = 'Load'
a = InitializeGraph(5)
GenerateTopology(a)
print(a)