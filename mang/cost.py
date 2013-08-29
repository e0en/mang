import numpy as np
import cudamat.gnumpy as gnp

# cost functions and -gradient of them


def squared_error(predicted, target):
    tmp = (target - predicted)
    return 0.5 * (tmp ** 2).sum()


def d_squared_error(predicted, target):
    return target - predicted


def cross_entropy(predicted, target):
    if type(predicted) == gnp.garray:
        log = gnp.log
    else:
        log = np.log
    return -(target * log(predicted + 1e-6) +
            (1. - target) * log(1. - predicted + 1e-6).sum())


def d_cross_entropy(predicted, target):
    return target / (predicted + 1e-6) - \
        (1. - target) * (1. - predicted + 1e-6)


cost_table = {
    "squared_error": squared_error,
    "cross_entropy": cross_entropy,
    }


d_cost_table = {
    "squared_error": d_squared_error,
    "cross_entropy": d_cross_entropy,
    }
