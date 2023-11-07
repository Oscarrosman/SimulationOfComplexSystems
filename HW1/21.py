import numpy as np
import matplotlib as plt

kb = 1.3806452*(10**-23)
T = 283
energyBarriers = [0, 2*kb*T, 0]
temp = []
probabilities = [[0, 0, 0] for i in range(3)]
generations = 10**5
state = [0]

for i in range(len(energyBarriers)):
    temp.append(np.exp(-energyBarriers[i]/(kb*T)))

probabilities[0][0] = temp[0]/(temp[0] + temp[1])
probabilities[0][1] = temp[1]/(temp[0] + temp[1])
probabilities[1][0] = temp[0]/sum(temp)
probabilities[1][1] = temp[1]/sum(temp)
probabilities[1][2] = temp[2]/sum(temp)
probabilities[2][1] = temp[1]/(temp[1] + temp[2])
probabilities[2][2] = temp[2]/(temp[1] + temp[2])

print(probabilities)

for iGen in range(generations - 1):
    r = np.random.rand()
    if state[-1] == 0:
        if r < probabilities[0][0]:
            state.append(0)
        else:
            state.append(1)
    elif state[-1] == 1:
        if r < probabilities[1][0]:
            state.append(0)
        elif r > (probabilities[1][0] + probabilities[1][1]):
            state.append(2)
        else:
            state.append(1)
    else:
        if r < probabilities[2][2]:
            state.append(2)
        else:
            state.append(1)

print(len(state))


