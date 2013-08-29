from operator import mul

import numpy as np
import cudamat.gnumpy as gnp


class Node(object):
    """Node with linear(or identity) activation function."""
    def __init__(self, shape, option):
        self.shape = shape
        self.size = reduce(mul, self.shape)
        self.b = np.zeros(self.size)
        self.on_gpu = False

    def init_training(self):
        self.b = gnp.garray(self.b)
        self.on_gpu = True

    def finish_training(self):
        self.b = self.b.asarray()
        self.on_gpu = False

    def up(self, x):
        if isinstance(x, np.ndarray) and self.on_gpu:
            b = self.b.asarray()
        else:
            b = self.b
        return x + b

    def down(self, y, dy):
        dy *= y*0. + 1.

    def gradient(self, dy):
        return dy.mean(0)
