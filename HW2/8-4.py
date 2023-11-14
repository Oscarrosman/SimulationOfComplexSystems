import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def InitializeParticles(l, n):
    positions = l*np.random.rand(n, 2)
    theta = 2*np.pi*np.random.rand(n,1)
    return positions, theta

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

def FindNeighbors(positions, rf, size):
    distances = [[] for _ in range(len(positions))]
    neighbors = [[] for _ in range(len(positions))]

    # In-line functions for different distance functions
    EuclidianDistance = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    WrappedDistance = lambda p1, p2, l: np.sqrt((l - abs(p1[0] - p2[0]))**2 + (l - abs(p1[1] - p2[1]))**2)

    # Calculate distances to other particles (Euclidian and wrapped)
    for i, x in enumerate(positions):
        for j, y in enumerate(positions):
            dist = [EuclidianDistance(x, y), WrappedDistance(x, y, size)]
            distances[i].append(min(dist))

    # Determine if other particles are within radius
    for i in range(len(distances)):
        for j in range(len(distances[0])):
            if distances[i][j] < rf:
                neighbors[i].append(j)

    return neighbors

def OrientationUpdate(angles, neigborhood, eta, dt):
    for i, neighbors in enumerate(neigborhood):
        thetas = [angles[a] for _, a in enumerate(neighbors)]
        avgSin = np.mean([np.sin(thetaK) for thetaK in thetas])
        avgCos = np.mean([np.cos(thetaK) for thetaK in thetas])
        avgTheta = np.arctan(avgSin/avgCos)
        w = np.random.uniform(-1/2, 1/2)
        angles[i] = avgTheta + eta*w*dt
    return angles

def UpdatePositions(positions, v, angle, size):
    for i in range(len(positions)):
        positions[i][0] += v*np.cos(angle[i])
        positions[i][1] += v*np.sin(angle[i])
        
        # Ensure particle stays within grid (If it moves outside it placed on the other side (Wraparound))
        if positions[i, 0] > size:
            positions[i, 0] -= size
        elif positions[i, 0] < size:
            positions[i, 0] += size
        
        if positions[i, 1] > size:
            positions[i, 1] -= size
        elif positions[i, 1] < size:
            positions[i, 1] += size
    return positions

def UpdateParticles(positions, angles, rf, l, eta, dt, v):
    neighbors = FindNeighbors(positions, rf, l)
    angles = OrientationUpdate(angles, neighbors, eta, dt)
    positions = UpdatePositions(positions, v, angles, l)
    return positions, angles

def VicsekModel(gen, v, size, mode='Load'):

    if mode == 'Load':
        particles = [LoadData('Positions.csv'), LoadData('Angles.csv')]
    else:
        particles = InitializeParticles(l, n)

    for i in range(gen):
        particles = UpdateParticles(particles[0], particles[1], rf, size, eta, dt, v)

        if i % 10 == 0:
            print(f'Generation: {i}')

    PlotFunctionA(particles[0])




l = 100
n = 100
v = 1
dt = 1
eta = 0.01
rf = 1
gen = 10**2

VicsekModel(gen, v, l)


