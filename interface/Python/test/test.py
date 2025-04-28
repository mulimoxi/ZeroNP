from pyzeronp import ZERONP
import numpy as np


# def cost(x: np.ndarray, par: int):
def cost(x: np.ndarray):
    # x is a vector of length 2
    f1 = (x[0]-5)**2 + x[1]**2-25
    f2 = -x[0]**2 + x[1]
    f = np.array([f1, f2])

    return f


p0 = [4.9, .1]
ib0 = [1]
ibl = [0]

prob = {'p0': p0, 'ib0': ib0, 'ibl': ibl}

myzeronp = ZERONP(prob=prob, cost=cost)
solution = myzeronp.run()
print(solution)
