import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def InitializeParticles(l, v, n):
    positions = l*np.random.rand(n, 2)
    velocities = v*np.random.rand(n, 2)
    return positions, velocities

def SaveData(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
def LoadData(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.split(',')
            line = [float(l) for l in line]
            data.append(line)
    return np.array(data)

def PlotFunctionA(data):

    fig, ax = plt.subplots()

    # Scatter plot of positions
    positions = [[data[j][i] for j in range(len(data))] for i in range(len(data[0]))]
    ax.scatter(positions[0], positions[1], s=10, color='b')

    # Add voroni tesselation
    vor = Voronoi(data)
    voronoi_plot_2d(vor, line_colors='k', line_width=1, show_vertices=False, line_alpha=0.5, point_size=0, ax=ax)
    ax.set_title('Initial positions')
    ax.set_xlabel('X: L')
    ax.set_ylabel('Y: L')
    plt.show()

def FindNeighbors(positions, rf):
    distances = [[] for _ in range(len(positions))]
    neighbors = [[] for _ in range(len(positions))]

    for i, x in enumerate(positions):
        for j, y in enumerate(positions):
            dist = [(x[k] - y[k])**2 for k in range(len(x))]
            distances[i].append(np.sqrt(sum(dist)))

    for i in range(len(distances)):
        for j in range(len(distances[0])):
            if i != j and distances[i][j] < rf:
                neighbors[i].append(j)

    return neighbors

def CalculateFlow(velocities, neighborList):
    # Find average flow in neighborhoods:
    for i in range(len(neighborList)):
        if neighborList[i]:
            avgV = [0 for _ in range(len(velocities[0]))]
            temp = [[] for _ in range(len(velocities[0]))]
            for j in range(len(neighborList[i])):
                for k in range(len(velocities[0])):
                    idx = neighborList[i][j]
                    temp[k].append(velocities[idx][k])


    pass



def VicsekModel(positions, velocities, rf, generations):
    for _ in range(generations):
        pass







# Variables

l = 100
n = 100
v = 1
dt = 1
eta = 0.01
rf = 1
gen = 10**4

testP, testV = InitializeParticles(l, v, n)
SaveData(testP, 'Positions.csv')
SaveData(testV, 'Velocities.csv')
testP = LoadData('Positions.csv')

d = FindNeighbors(testP, rf=1)
print(d)
