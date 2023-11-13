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

def PlotFunction(data):
    colors = ['red', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    counter = 0
    titles = ['Steps = 0', 'Steps = 100', 'Steps = 500', 'Steps = 1000']

    for step in range(len(data)):
        counter += 1
        plt.subplot(3, 4, counter)
        plt.imshow(data[step][0], cmap=cmap, interpolation='nearest')
        plt.title(titles[step])
        if counter == 1:
            plt.ylabel('T = Tcrit')
        elif counter == 4:
            plt.colorbar()
    
    for step in range(len(data)):
        counter += 1
        plt.subplot(3, 4, counter)
        plt.imshow(data[step][1], cmap=cmap, interpolation='nearest')
        if counter == 5:
            plt.ylabel('T = 5 (>Tcrit)')
        elif counter == 8:
            plt.colorbar()

    for step in range(len(data)):
        counter += 1
        plt.subplot(3, 4, counter)
        plt.imshow(data[step][2], cmap=cmap, interpolation='nearest')
        if counter == 9:
            plt.ylabel('T = 10 (>Tcrit)')
        elif counter == 12:
            plt.colorbar()

    plt.suptitle('All latices')
    plt.tight_layout()
    plt.show()

def PlotFunctionD(data, H):
     
     # Find coefficient X with linear regression
     coefficients = np.polyfit(H, data, 1)
     x = coefficients[0]
     print(f'\nCoefficient (X) approximated to: {x} (m = xH)')
     
    # Plot of m as a function of H
     #plt.plot(H, data, linestyle='-', color='r')

    # Plot of m as a function of H
     plt.plot(H, data, linestyle='-', color='r', label='Data Points')
    
    # Plot the line y = kx with black color
     plt.plot(H, x * np.array(H), linestyle='-', color='k', label=f'Fit: m = {x:.2f}H')
     plt.xlabel('External magnetization, H')
     plt.ylabel('Internal magnetization, m(H)')
     plt.grid(True)
     plt.xlim(0, max(H))
     plt.ylim(0, max(data)+0.25*max(data))
     plt.legend()
     plt.show()

def Magnetization(grid):
    N = len(grid)*len(grid[0])
    m = (1/(N**2))*sum(sum(grid))
    return m

def IsingMethodABC(temp, steps, size):
    grids = [Initialization(size, size) for _ in range(len(temp))]
    plotData = [[], [], [], []] 
    st = time.time()
    for step in range(steps):
        for i, t in enumerate(temp):
            grids[i] = MonteCarloStep(grids[i], t, H)

        if step == 0:
            plotData[0] = copy.deepcopy(grids)
        if step == 100:
            plotData[1] = copy.deepcopy(grids)
        if step == 500:
            plotData[2] = copy.deepcopy(grids)
        if step == 1000:
            plotData[3] = copy.deepcopy(grids)

        if step % 500 == 0:
            print(f'Steps taken: {step}, ({100*step/steps:3.1f}%)')
    print(f'\nSimulation time: {(time.time()-st)//60:2.0f} min, {(time.time()-st)%60:2.0f} seconds.\n')
    PlotFunction(plotData)

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


# Part A
H = 0
J = 1
kb = 1
tCrit = 2.269
temperatures = [1, 3, 6]
#IsingMethodABC(temperatures, 10001, 200)

# Part B
temperatures = [0.25*tCrit, tCrit, 2*tCrit]
IsingMethodABC(temperatures, 1001, 200)

# Part C
temperatures = [tCrit, 5, 10]
#IsingMethodABC(temperatures, 10001, 200)

# Part D
H = np.linspace(0, 0.02, num=50)
#IsingMethodD(tCrit, 1000, 200, H)



