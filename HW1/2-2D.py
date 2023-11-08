import numpy as np
import matplotlib.pyplot as plt
import copy, time

def Initialization(m, n):
    return np.random.choice([-1, 1], size=(m, n))

def MonteCarloStep(lattice, T, h):
    beta = 1/T # beta = T^-1 according to book

    # List of linear indicies
    indicies = [i for i in range(len(lattice)*len(lattice[0]))]

    # Scramble order of indicies
    indicies = np.random.permutation(indicies)

    # Number of changes made (10%)
    changes = int(0.1*len(lattice)*len(lattice[0]))

    # Calculate sum and change the cell with probability for 10% of cells

    for cell in range(changes):
        M = 0
        i = indicies[cell] // len(lattice[0])
        j = indicies[cell] % len(lattice[0])

        if i > 0:
            M += lattice[i-1][j]
        if j > 0:
            M += lattice[i][j-1]
        if i < len(lattice)-1:
            M += lattice[i+1][j]
        if j < len(lattice[0]) -1:
            M += lattice[i][j+1]

        # Calculate energy of surroundings
        energies = [-(h+J*M), h+J*M] # [+, -]
        p = [np.exp(-energies[1]/T), np.exp(-energies[0]/T)]
        probabilities = [p[0]/sum(p), p[1]/sum(p)]

        # Set spin based on probability
        lattice[i][j] = np.random.choice([-1, 1], p=probabilities)
    
    return lattice

def PlotFunctionD(data, H):
     
     # Find coefficient X with linear regression
     mValues = np.array(data)
     HValues = np.array(data)
     x = np.sum(mValues*HValues)/np.sum(HValues**2)
     print(f'\nCoefficient (X) approximated to: {x} (m = xH)')
     
    # Plot of m as a function of H
     plt.plot(H, data, linestyle='-', color='r')
     plt.xlabel('External magnetization, H')
     plt.ylabel('Internal magnetization, m(H)')
     plt.grid(True)
     plt.xlim(0, max(H))
     plt.ylim(0, max(data)+0.25*max(data))
     plt.show()

def Magnetization(grid):
    N = len(grid)*len(grid[0])
    m = (1/(N**2))*sum(sum(grid))
    return m

def IsingMethodD(temp, steps, size, hList):
    grid = Initialization(size, size)
    plotData = [] 
    st = time.time()

    for i,h in enumerate(hList):
        for step in range(steps):
            grid = MonteCarloStep(grid, temp, h)

        m = Magnetization(grid)
        plotData.append(m)

        print(f'{i}) m(H = {h}) = {m}')

    print(f'\nSimulation time: {(time.time()-st)//60:2.0f} min, {(time.time()-st)%60:2.0f} seconds.\n')
    PlotFunctionD(plotData, hList)


# Variables
H = [0, 2, 4, 6, 8, 10] # External magnetic field
H = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
J = 1
kb = 1
tCrit = 2.269

IsingMethodD(tCrit, 1000, 200, H)