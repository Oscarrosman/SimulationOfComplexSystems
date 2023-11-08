import numpy as np
import matplotlib.pyplot as plt
import time

def SimulationAB(eb, t, steps=10**5):
    states = ['Left', 'Middle', 'Right']
    currentState = 'Left'
    transitions = {'Left': 0, 'Middle': 0, 'Right': 0}

    # Sets probability of states and energy levels
    E = [0, eb, 0]
    p = [np.exp(-i/(kb*t)) for i in E]
    sums = [p[0]+p[1], p[0]+p[1]+p[2], p[1]+p[2]]
    distribution = None
    prevDistribution = None
    threshold = 10**-4

    for step in range(steps):
        # Sets probabilities from current position
        if currentState == 'Left':
            probabilities = [p[0]/sums[0], p[1]/sums[0], 0]
        elif currentState == 'Middle':
            probabilities = [p[0]/sums[1], p[1]/sums[1], p[2]/sums[1]]
        else:
            probabilities = [0, p[1]/sums[2], p[2]/sums[2]]

        # Chooses next move
        nextState = np.random.choice(states, p=probabilities)

        # Logs next move
        transitions[nextState] += 1
        currentState = nextState

        # Calculates distribution every 10 steps
        if step % 10 == 0 and step != 0:
            distribution = {state: count / step for state, count in transitions.items()}
            # Checks for equilibrium
            if prevDistribution is not None:
                change = max([abs(distribution[state] - prevDistribution[state]) for state in states])
                # Check if threshold is reached
                if change < threshold:
                    print(f'\nEquilibrium distribution reached at {step} with tolerance: {threshold}')
                    print(distribution)
                    prevDistribution = distribution
                    threshold *= 10**-1
                else:
                    prevDistribution = distribution
            else:
                prevDistribution = distribution

def SimulationC(eb, t, steps=10**5):
    # Mostly the same as Simulation AB but breaks when it reaches the right side
    states = ['Left', 'Middle', 'Right']
    currentState = 'Left'
    history = ['Left']
    transitions = {'Left': 0, 'Middle': 0, 'Right': 0}
    E = [0, eb, 0]
    p = [np.exp(-i/(kb*t)) for i in E]
    sums = [p[0]+p[1], p[0]+p[1]+p[2], p[1]+p[2]]
    gen = 0

    for step in range(steps):
        gen += 1
        if currentState == 'Left':
            probabilities = [p[0]/sums[0], p[1]/sums[0], 0]
        elif currentState == 'Middle':
            probabilities = [p[0]/sums[1], p[1]/sums[1], p[2]/sums[1]]
        else:
            return gen

        nextState = np.random.choice(states, p=probabilities)
        transitions[nextState] += 1
        history.append(nextState)
        currentState = nextState
    

# Part A: Eb = 2kBT

t = 1
kb = 1.380649*10**-23
eb = 2*kb*t

SimulationAB(eb, t)

# Part B: Varying Eb and T

T = [0.5, 1, 2, 4, 8]
Eb = [1.25*kb]
genT = 0
genEB = 0

for t in T:
    genT += 1
    genEB = 0
    for eb in Eb:
            genEB +=1
            print(f'Simulation: {genT}.{genEB}')
            print(f' > Eb = {eb/kb}kbT \n > T = {t}')
            SimulationAB(eb*t, t)
            print('\n \n')

# Part C: 

kb = 1.380649*10**-23
Eb = [0.5*kb, 1*kb, 2*kb, 4*kb]
genT = 0
genEB = 0
st = time.time()
for t in T:
    genT += 1
    genEB = 0
    for eb in Eb:
        genEB += 1
        escapeTime = []
        for i in range(1000):
            eT = SimulationC(eb, t)
            escapeTime.append(eT)
        escapeTime = np.array(escapeTime)
        print(f'\n\nSimulation: {genT}.{genEB}')
        print(f' > Eb = {eb/kb}kbT \n > T = {t}')
        print(f'Average escape time: {np.mean(escapeTime)}')

print(f'\nSimulation time: {(time.time()-st)//60:2.0f} min, {(time.time()-st)%60:2.0f} seconds.\n')

