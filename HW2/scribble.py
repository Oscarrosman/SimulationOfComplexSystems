import numpy as np

def PositionUpdate(x, l, v, dt):
    for i, angV in enumerate(v):
        x[i] += angV*dt
        if x[i] > l/2:
            x[i] -= l
        elif x[i] < -l/2:
            x[i] += l
    return x

def VelocityUpdate(x, v, l):
    pass
