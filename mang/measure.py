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
    n_sample = predicted.shape[0]
    target_list = [set(x.nonzero()[0]) for x in target]
    predicted_list = [set(x.argsort()[:k]) for x in -predicted]
    divisor = [max(min(k, len(x)), 1) for x in target_list]
    n_correct = \
        [len(target_list[i] & predicted_list[i]) for i in xrange(n_sample)]
    return np.mean([1. * n_correct[i] / divisor[i] for i in xrange(n_sample)])


def recall(predicted, target, k=5):
    n_sample = predicted.shape[0]
    target_list = [set(x.nonzero()[0]) for x in target]
    predicted_list = [set(x.argsort()[:k]) for x in -predicted]
    divisor = [max(len(x), 1) for x in target_list]
    n_correct = \
        [len(target_list[i] & predicted_list[i]) for i in xrange(n_sample)]
    return np.mean([1. * n_correct[i] / divisor[i] for i in xrange(n_sample)])


def f1_score(predicted, target, k=5):
    n_sample = predicted.shape[0]
    target_list = [set(x.nonzero()[0]) for x in target]
    predicted_list = [set(x.argsort()[:k]) for x in -predicted]
    n_correct = \
        [len(target_list[i] & predicted_list[i]) for i in xrange(n_sample)]

    divisor = [max(min(k, len(x)), 1) for x in target_list]
    p_list = \
        np.array([1. * n_correct[i] / divisor[i] for i in xrange(n_sample)])

    divisor = [max(len(x), 1) for x in target_list]
    r_list = \
        np.array([1. * n_correct[i] / divisor[i] for i in xrange(n_sample)])

    divisor = [max(1e-6, x) for x in p_list + r_list]

    return (2 * p_list * r_list / divisor).mean()


def annotate_as_matrix(predicted, k):
    (n_sample, n_tag) = predicted.shape

    # annotate images with top-k tags
    predicted_matrix = np.zeros(predicted.shape)
    for i in xrange(n_sample):
        idx = (-predicted[i]).argsort()[:k]
        predicted_matrix[i, idx] = 1
    return predicted_matrix


def tag_precision(predicted, target, k=5):
    (n_sample, n_tag) = predicted.shape
    target_matrix = 1. * (target > 0)
    predicted_matrix = annotate_as_matrix(predicted, k)

    # for each tags, retrieve images and calculate precision
    correct_matrix = target_matrix * predicted_matrix
    precisions = np.zeros((n_tag, ))
    n_plus = 0
    for j in xrange(n_tag):
        n_retrieved = predicted_matrix[:, j].sum()
        if n_retrieved > 0:
            n_correct = correct_matrix[:, j].sum()
            precisions[j] = n_correct / n_retrieved
            n_plus += 1

    return (precisions.sum() / n_plus, n_plus)


def tag_recall(predicted, target, k=5):
    (n_sample, n_tag) = predicted.shape
    target_matrix = 1. * (target > 0)
    predicted_matrix = annotate_as_matrix(predicted, k)

    # for each tags, retrieve images and calculate precision
    correct_matrix = target_matrix * predicted_matrix
    recalls = np.zeros((n_tag, ))
    for j in xrange(n_tag):
        n_truth = target_matrix[:, j].sum()
        n_correct = correct_matrix[:, j].sum()
        recalls[j] = n_correct / n_truth

    return recalls.mean()


def tag_f1_score(predicted, target, k=5):
    (n_sample, n_tag) = predicted.shape
    target_matrix = 1. * (target > 0)
    predicted_matrix = annotate_as_matrix(predicted, k)

    # for each tags, retrieve images and calculate f1-score
    correct_matrix = target_matrix * predicted_matrix
    f1_scores = np.zeros((n_tag, ))
    n_plus = 0

    for j in xrange(n_tag):
        n_retrieved = predicted_matrix[:, j].sum()
        if n_retrieved > 0:
            n_correct = correct_matrix[:, j].sum()
            p = n_correct / n_retrieved
            n_truth = target_matrix[:, j].sum()
            n_correct = correct_matrix[:, j].sum()
            r = n_correct / n_truth
            if p * r == 0:
                n_plus += 1
            elif p + r > 0:
                f1_scores[j] = 2 * p * r / (p + r)
                n_plus += 1

    return (f1_scores.sum() / n_plus, n_plus)


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
    "tag_precision": tag_precision,
    "tag_recall": tag_recall,
    "tag_f1_score": tag_f1_score,
    }
