from operator import mul

import numpy as np

import mang.cudamat as cm


class Edge(object):
    def __init__(self, shape_in, shape_out, option={}):
        size_in = reduce(mul, shape_in)
        size_out = reduce(mul, shape_out)
        self.shape = (size_in, size_out)
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.W = np.zeros(self.shape)
        self.on_gpu = False

    def init_training(self, option):
        self.W = cm.CUDAMatrix(self.W)
        self.on_gpu = True

    def finish_training(self):
        W = self.W.asarray()
        self.W.free_device_memory()
        self.W = W
        self.on_gpu = False

    def up(self, x, o=None):
        if o is None:
            W = self.W.asarray() if self.on_gpu else self.W
            return np.dot(x, W)
        else:
            o.add_dot(x, self.W)

    def down(self, do, dx, o, x):
        dx.add_dot(do, self.W.T)

    def gradient(self, x, do, gW):
        gW.add_dot(x.T, do, 1. / x.shape[0])
