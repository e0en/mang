import numpy as np


def sse(predicted, target, **option):
    return ((predicted - target) ** 2).sum()


def mse(predicted, target, **option):
    return ((predicted - target) ** 2).mean()


def rmse(predicted, target, **option):
    return np.sqrt(((predicted - target) ** 2).mean(1)).mean()


def accuracy(predicted, target, **option):
    label1 = np.argmax(predicted, 1)
    label2 = np.argmax(target, 1)
    return (label1 == label2).mean()


def hamming(predicted, target, **option):
    return (predicted != target).mean()


def one_error(predicted, arget, **option):
    pass


def average_precision(predicted, target, **option):
    pass


def precision_at_k(k):
    return lambda predicted, target: precision(predicted, target, k)


def recall_at_k(k):
    return lambda predicted, target: recall(predicted, target, k)


def precision(predicted, target, k=5):
    N = predicted.shape[0]
    target_list = [set(x.nonzero()[0]) for x in target]
    predicted_list = [set(x.argsort()[:k]) for x in -predicted]
    divisor = [max(min(k, len(x)), 1) for x in target_list]
    n_correct = [len(target_list[i] & predicted_list[i]) for i in xrange(N)]
    return np.mean([1. * n_correct[i] / divisor[i] for i in xrange(N)])


def recall(predicted, target, k=5):
    N = predicted.shape[0]
    target_list = [set(x.nonzero()[0]) for x in target]
    predicted_list = [set(x.argsort()[:k]) for x in -predicted]
    divisor = [max(len(x), 1) for x in target_list]
    n_correct = [len(target_list[i] & predicted_list[i]) for i in xrange(N)]
    return np.mean([1. * n_correct[i] / divisor[i] for i in xrange(N)])


def f1_score(predicted, target, k=5):
    N = predicted.shape[0]
    target_list = [set(x.nonzero()[0]) for x in target]
    predicted_list = [set(x.argsort()[:k]) for x in -predicted]
    n_correct = [len(target_list[i] & predicted_list[i]) for i in xrange(N)]

    divisor = [max(min(k, len(x)), 1) for x in target_list]
    p_list = np.array([1. * n_correct[i] / divisor[i] for i in xrange(N)])

    divisor = [max(len(x), 1) for x in target_list]
    r_list = np.array([1. * n_correct[i] / divisor[i] for i in xrange(N)])

    divisor = [max(1e-6, x) for x in p_list + r_list]

    return (2 * p_list * r_list / divisor).mean()


MEASURE_TABLE = {
    "sse": sse,
    "mse": mse,
    "rmse": rmse,
    "accuracy": accuracy,
    "hamming": hamming,
    "one_error": one_error,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    }
