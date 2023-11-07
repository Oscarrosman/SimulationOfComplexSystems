import numpy as np
import matplotlib.pyplot as plt

def simulationAB(eb, t, steps=10**5):
    states = ['Left', 'Middle', 'Right']
    currentState = 'Left'
    history = ['Left']
    transitions = {'Left': 0, 'Middle': 0, 'Right': 0}
    E = [0, eb, 0]
    p = [np.exp(-i/(kb*t)) for i in E]
    sums = [p[0]+p[1], p[0]+p[1]+p[2], p[1]+p[2]]
    distribution = None
    prevDistribution = None
    threshold = 10**-4

    for step in range(steps):
        if currentState == 'Left':
            probabilities = [p[0]/sums[0], p[1]/sums[0], 0]
        elif currentState == 'Middle':
            probabilities = [p[0]/sums[1], p[1]/sums[1], p[2]/sums[1]]
        else:
            probabilities = [0, p[1]/sums[2], p[2]/sums[2]]

        nextState = np.random.choice(states, p=probabilities)
        transitions[nextState] += 1
        history.append(nextState)
        currentState = nextState

        if step % 10 == 0 and step != 0:
            distribution = {state: count / step for state, count in transitions.items()}
            if prevDistribution is not None:
                change = max([abs(distribution[state] - prevDistribution[state]) for state in states])
                #change = max([abs(distribution[state] - prevDistribution[state] for state in states)])
                if change < threshold:
                    print(f'Equilibrium distribution reached at {step}')
                    print(distribution)
                    prevDistribution = distribution
                    threshold *= 10**-1
                else:
                    prevDistribution = distribution
            else:
                prevDistribution = distribution

def simulationC(eb, t, steps=10**5):
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
            #print(f'Escape time: {step}')
            return gen

        nextState = np.random.choice(states, p=probabilities)
        transitions[nextState] += 1
        history.append(nextState)
        currentState = nextState
    




    
    



        

# Part A: Eb = 2kBT

t = 1
kb = 1.380649*10**-23
eb = 2*kb*t

#simulation(eb, t)

# Part B: Varying Eb and T

T = [0.5, 1, 2, 4]
Eb = [[0.5*kb*t for t in T], [1*kb*t for t in T], [2*kb*t for t in T], [4*kb*t for t in T]] # 4^2 (16) different variations of eb and T
genT = 0
genEB = 0

for t in T:
    genT += 1
    genEB = 0
    for l in Eb:
        for eb in l:
            genEB +=1
            print(f'Simulation: {genT}.{genEB}')
            print(f' > Eb = {eb} \n > T = {t}')
            #simulationAB(eb, t)
            print('\n \n')

# To do: reach conclusion regarding equilibriums

# Part C: 

t = 1
kb = 1.380649*10**-23
eb = 2*kb*t
escapeTime = []
for i in range(1000):
    eT = simulationC(eb, t)
    escapeTime.append(eT)

print(f'Average escape time: {np.mean(escapeTime)}')



