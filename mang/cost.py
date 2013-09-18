import numpy as np

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


def d_l1_norm(predicted, result, scale=0.1):
    predicted.sign(result)
    result.mult(-scale)


COST_TABLE = {
    'squared_error': squared_error,
    'cross_entropy': cross_entropy,
    }


UNSUPERVISED_D_COST_TABLE = {
    'l1_norm': d_l1_norm,
    }


D_COST_TABLE = {
    'squared_error': d_squared_error,
    'cross_entropy': d_cross_entropy,
    }
