import numpy as np


def sse(y, target, **option):
    return ((y - target)**2).sum()

def mse(y, target, **option):
    return ((y - target)**2).mean()

def rmse(y, target, **option):
    return np.sqrt(((y - target)**2).mean(1)).mean()

def accuracy(y, target, **option):
    label1 = np.argmax(y, 1)
    label2 = np.argmax(target, 1)
    return (label1 == label2).mean()

def hamming_loss(y, target, **option):
    return (y != target).mean()

def ranking_loss(y, arget, **option):
    pass

def one_error(y, arget, **option):
    pass

def average_precision(y, target, **option):
    pass

def precision_at_k(k):
    return lambda y, target: precision(y, target, k)

def recall_at_k(k):
    return lambda y, target: recall(y, target, k)

def precision(y, target, k):
    N = y.shape[0]
    target_list = [set(x.nonzero()[0]) for x in target]
    y_list = [set(x.argsort()[:k]) for x in -y]
    divisor = [max(min(k, len(x)), 1) for x in target_list]
    n_correct = [len(target_list[i] & y_list[i]) for i in xrange(N)]
    return np.mean([1.*n_correct[i]/divisor[i] for i in xrange(N)])

def recall(y, target, k):
    N = y.shape[0]
    target_list = [set(x.nonzero()[0]) for x in target]
    y_list = [set(x.argsort()[:k]) for x in -y]
    divisor = [max(len(x), 1) for x in target_list]
    n_correct = [len(target_list[i] & y_list[i]) for i in xrange(N)]
    return np.mean([1.*n_correct[i]/divisor[i] for i in xrange(N)])
