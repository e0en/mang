import numpy as np
import mang.cudamat as cm

# cost functions and -gradient of them


def squared_error(predicted, target):
    tmp = (target - predicted)
    return 0.5 * (tmp ** 2).sum()


def d_squared_error(predicted, target, result):
    target.subtract(predicted, result)


def cross_entropy(predicted, target):
    return -(target * np.log(predicted + 1e-6) +
            (1. - target) * np.log(1. - predicted + 1e-6).sum())


def d_cross_entropy(predicted, target, result):
    raise NotImplementedError
    return target / (predicted + 1e-6) - \
        (1. - target) * (1. - predicted + 1e-6)


COST_TABLE = {
    "squared_error": squared_error,
    "cross_entropy": cross_entropy,
    }


D_COST_TABLE = {
    "squared_error": d_squared_error,
    "cross_entropy": d_cross_entropy,
    }
