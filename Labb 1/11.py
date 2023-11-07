import numpy as np
import matplotlib as plt


r0 = 0
v0 = 0
m = 0
k = 1
timeStep = 0.1

A = np.sqrt((r0**2) + (m/k)*(v0**2))
theta = np.arccos(r0/A)
w = np.sqrt(k/m)

position = []
velocity = []
mechEnergy = []
FindEnergy = lambda x, v, k, m: ((0.5*k*x**2) + (0.5*m*v**2))


# start while/ for loop

pNew = position[-1] + velocity[-1]*timeStep
force = -k*pNew
vNew = velocity[-1] + (force/m)*timeStep
position.append(pNew)
velocity.append(vNew)
mechEnergy.append(FindEnergy(position[-1], velocity[-1], k, m))