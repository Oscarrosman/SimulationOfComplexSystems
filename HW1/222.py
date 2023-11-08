import numpy as np
import matplotlib.pyplot as plt
import copy
import time

def Initialization(m, n):
    return np.random.choice([-1, 1], size=(m, n))

def MonteCarloStep(lattice, T):
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
        energies = [-(H+J*M), H+J*M] # [+, -]
        p = [np.exp(-energies[1]/T), np.exp(-energies[0]/T)]
        probabilities = [p[0]/sum(p), p[1]/sum(p)]

        # Set spin based on probability
        lattice[i][j] = np.random.choice([-1, 1], p=probabilities)
    
    return lattice

def PlotFunction(data):
    colors = ['red', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    counter = 0
    titles = ['Steps = 0', 'Steps = 100', 'Steps = 1000', 'Steps = 10,000']

    for step in range(len(data)):
        counter += 1
        plt.subplot(3, 4, counter)
        plt.imshow(data[step][0], cmap=cmap, interpolation='nearest')
        plt.title(titles[step])
        if counter == 1:
            plt.ylabel('T < Tcrit')
    
    for step in range(len(data)):
        counter += 1
        plt.subplot(3, 4, counter)
        plt.imshow(data[step][1], cmap=cmap, interpolation='nearest')
        if counter == 5:
            plt.ylabel('Tcrit')

    for step in range(len(data)):
        counter += 1
        plt.subplot(3, 4, counter)
        plt.imshow(data[step][2], cmap=cmap, interpolation='nearest')
        if counter == 9:
            plt.ylabel('T > Tcrit')

    plt.suptitle('All latices')
    plt.tight_layout()
    plt.show()

def IsingMethod(temp, steps, size):
    grids = [Initialization(size, size) for _ in range(len(temp))]
    plotData = [[], [], [], []] 
    st = time.time()
    for step in range(steps):
        for i, t in enumerate(temp):
            grids[i] = MonteCarloStep(grids[i], t)

        if step == 0:
            plotData[0] = copy.deepcopy(grids)
        elif step == 100:
            plotData[1] = copy.deepcopy(grids)
        elif step == 1000:
            plotData[2] = copy.deepcopy(grids)
        elif step == 10000:
            plotData[3] = copy.deepcopy(grids)

        if step % 500 == 0:
            print(f'Steps taken: {step}, ({100*step/steps:3.1f}%)')
    print(f'Simulation time: {(time.time()-st)//60}min, {(time.time()-st)%60} seconds.')
    PlotFunction(plotData)


# Variables
H = 0
J = 1
kb = 1
tCrit = 2.269
temperatures = [0.25*tCrit, tCrit, 2*tCrit]

IsingMethod(temperatures, 10001, 200)

test = np.random.choice([1, 1], size=(20, 20))
print(test)
test = MonteCarloStep(test, 0.01*tCrit)
counter = 0
for i in range(len(test)):
    for j in range(len(test[0])):
        if test[i][j] == -1:
            counter +=1
print(f'Number of changes: {counter}')
print(test)


