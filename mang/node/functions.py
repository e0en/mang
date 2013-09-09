import numpy as np


def logistic(x):
    return 1./(1. + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    tmp = x.T - x.max(1) # for numerical stability
    tmp = np.exp(tmp)
    return (tmp/(tmp.sum(0) + 0.01)).T

def inv_cubic(x):
    y = 0.*x
    y_now = y + 1.
    while abs(y_now - y).max() > 1e-6:
        y_now = y
        y = (2./3.*y_now**3 + x)/(y_now**2 + 1)
    return y
