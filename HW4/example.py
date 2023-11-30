import numpy as np
import matplotlib.pyplot as plt

def Initialization(n, noise):
    # Create perfect grid
    x = np.linspace(0,1, num=n)
    y = np.linspace(0,1, num=n)
    nodes = [[i, j] for i in x for j in y]

    # Introduce noise
    for i in range(n**2):
        nodes[i][0] += noise*np.random.uniform(-1, 1)
        nodes[i][1] += noise*np.random.uniform(-1, 1)

    # Create connections
    # Initialize connections to form squares
    connections = np.zeros(shape=(n**2, n**2))
    for i in range(len(connections)):
        if i % n == 0:
            # Left side
            if i // n == 0:
                # Top
                connections[i][i+1] = 1 # Right
                connections[i][i+n] = 1 # Below
            elif i // n == (n-1):
                # bottom
                connections[i][i+1] = 1 # Right
                connections[i][i-n] = 1 # Above
            else:
                # Middle
                connections[i][i+1] = 1 # Right
                connections[i][i+n] = 1 # Below
                connections[i][i-n] = 1 # Above
            
        elif i % n == (n-1):
            # Right side
            if i // n == 0:
                # Top
                connections[i][i-1] = 1 # Left
                connections[i][i+n] = 1 # Below
            elif i // n == (n-1):
                # bottom
                connections[i][i-1] = 1 # Left
                connections[i][i-n] = 1 # Above
            else:
                # Middle
                connections[i][i-1] = 1 # Left
                connections[i][i+n] = 1 # Below
                connections[i][i-n] = 1 # Above
        else:
            # Internally
            if i // n == 0:
                # Top
                connections[i][i+1] = 1 # Right
                connections[i][i-1] = 1 # Left
                connections[i][i+n] = 1 # Below
            elif i // n == (n-1):
                # bottom
                connections[i][i+1] = 1 # Right
                connections[i][i-1] = 1 # Left
                connections[i][i-n] = 1 # Above
            else:
                # Middle
                connections[i][i+1] = 1 # Right
                connections[i][i-1] = 1 # Left
                connections[i][i-n] = 1 # Above
                connections[i][i+n] = 1 # Below
    np.fill_diagonal(connections, 0)
    return nodes, connections


def PlotFunction(nodes):
    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]

    plt.scatter(x, y)
    plt.show()

def PlotFunction1(nodes, adjacency):
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
        print(counter)
        counter += 1
        plt.plot(pair[0], pair[1], linewidth=0.1, color='blue')

    plt.xticks([]), plt.yticks([])
    plt.scatter(x, y, color='orange')
    plt.show()

nod, con = Initialization(10, 0.025)
PlotFunction1(nod, con)