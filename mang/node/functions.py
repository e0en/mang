import numpy as np
import cudamat.gnumpy as gnp

def logistic(x):
    if isinstance(x, gnp.garray):
        return 1./(1. + gnp.exp(-x))
    else:
        return 1./(1. + np.exp(-x))

def tanh(x):
    if isinstance(x, gnp.garray):
        return gnp.tanh(x)
    else:
        return np.tanh(x)

def softmax(x):
    tmp = x.T - x.max(1) # for numerical stability
    if isinstance(x, gnp.garray):
        tmp = gnp.exp(tmp)
    else:
        tmp = np.exp(tmp)
    return (tmp/(tmp.sum(0) + 0.01)).T

def inv_cubic(x):
    y = 0.*x
    y_now = y + 1.
    while abs(y_now - y).max() > 1e-6:
        y_now = y
        y = (2./3.*y_now**3 + x)/(y_now**2 + 1)
    return y
