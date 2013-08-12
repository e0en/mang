from operator import mul

import numpy as np
import cudamat.gnumpy as gnp


class Node(object):
    def __init__(self, shape, option):
        self.shape = shape
        self.size = reduce(mul, self.shape)
        self.b = np.zeros(self.size)

    def init_training(self, option):
        if option["reset"]:
            self.b = np.zeros(self.size)
        self.b_g = gnp.garray(self.b)
        self.db = gnp.zeros(self.b.shape)
        self.gb = gnp.zeros(self.b.shape)

    def copy_params(self):
        self.b = self.b_g.asarray()

    def finish_training(self):
        del self.b_g, self.db, self.gb

    def up(self, x):
        b = self.b_g if isinstance(x, gnp.garray) else self.b
        return x + b

    def down(self, y, dy):
        dy *= y*0. + 1.

    def gradient(self, dy):
        self.gb = dy.mean(0)

    def update(self, eps, momentum):
        self.db *= momentum
        self.db += eps * self.gb
        self.b_g += self.db
