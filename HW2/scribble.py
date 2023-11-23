import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

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


def OrientationUpdate(angles, neigborhood, eta, dt, history, h):
    if len(history) < h or h == 0:
        for i, neighbors in enumerate(neigborhood):
            thetas = [angles[a] for _, a in enumerate(neighbors)]
            avgSin = np.mean([np.sin(thetaK) for thetaK in thetas])
            avgCos = np.mean([np.cos(thetaK) for thetaK in thetas])
            avgTheta = np.arctan(avgSin/avgCos)
            w = np.random.uniform(-1/2, 1/2)
            angles[i] = avgTheta + eta*w*dt
    elif h < 0:
        pass
        for i, neighbors in enumerate(neigborhood):
            # Single out particle history
            thetas = [[history[i][k] for i in range(len(history))] for _, k in enumerate(neighbors)]
            x = np.linspace(1, abs(h), num=abs(h))
            # Determine trajectory
            coefficients = [np.polyfit(x, tH, 1) for tH in thetas]
            x1 = h+1
            # Extrapolate
            thetas1 = [np.polyval(c, x1) for c in coefficients]
            # Average trajectories
            avgSin = np.mean([np.sin(thetaK) for thetaK in thetas1])
            avgCos = np.mean([np.cos(thetaK) for thetaK in thetas1])
            avgTheta = np.arctan(avgSin/avgCos)
            # Randomize W
            w = np.random.uniform(-1/2, 1/2)
            angles[i] = avgTheta + eta*w*dt
    else:
        for i, neighbors in enumerate(neigborhood):
            thetas = [history[0][a] for _, a in enumerate(neighbors)]
            avgSin = np.mean([np.sin(thetaK) for thetaK in thetas])
            avgCos = np.mean([np.cos(thetaK) for thetaK in thetas])
            avgTheta = np.arctan(avgSin/avgCos)
            w = np.random.uniform(-1/2, 1/2)
            angles[i] = avgTheta + eta*w*dt
    return angles
