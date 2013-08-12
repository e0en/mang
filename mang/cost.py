import numpy as np
import cudamat.gnumpy as gnp

# cost functions and -gradient of them

def squared_error(predicted, target, is_d=False):
    tmp = (target- predicted)
    if is_d:
        return tmp
    else:
        return 0.5*(tmp**2).sum()

def cross_entropy(predicted, target, is_d=False):
    if is_d:
        return target/(predicted + 1e-6) -\
                (1. - target)*(1. - predicted + 1e-6)
    else:
        if type(predicted) == gnp.garray:
            log = gnp.log
        else:
            log = np.log
        return -(target*log(predicted + 1e-6) +\
                (1. - target)*log(1. - predicted + 1e-6).sum())

# this cost function is almost impossible to parallelize efficiently
def multi_label(predicted, target, is_d=False):
    if type(predicted) == gnp.garray:
        exp = gnp.exp
    else:
        exp = np.exp
    n_label = target.shape[1]
    denom = target.sum(1)
    max_card = denom.max()
    denom = denom*(n_label - denom)
    if is_d:
        raise NotImplementedError
    else:
        raise NotImplementedError

cost_table = {
        "squared_error": squared_error,
        "cross_entropy": cross_entropy,
        }
