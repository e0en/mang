import numpy as np

import mang.cudamat as cm
from mang.edge.edge import Edge


class EdgeRef(object):
    def __init__(self, original, option):
        self.original = original
        self.transpose = option["transpose"]

    def init_training(self, option):
        pass

    def finish_training(self):
        pass

    def up(self, x, o=None):
        if o is None:
            if self.original.on_gpu:
                W = self.original.W.asarray()
            else:
                W = self.original.W
            W = W.T if self.transpose else W
            return np.dot(x, W)
        else:
            if self.transpose:
                o.add_dot(x, self.original.W.T)
            else:
                o.add_dot(x, self.original.W)

    def down(self, do, dx, o, x):
        if self.transpose:
            dx.add_dot(do, self.original.W)
        else:
            dx.add_dot(do, self.original.W.T)

    def gradient(self, x, do, gW):
        if self.transpose:
            gW.add_dot(do.T, x, 1. / x.shape[0])
        else:
            gW.add_dot(x.T, do, 1. / x.shape[0])

    def materialize(self):
        if self.transpose:
            new_edge = Edge(self.original.shape_out, self.original.shape_in)
            new_edge.W = np.array(self.original.W)
        else:
            new_edge = Edge(self.original.shape_in, self.original.shape_out)
            new_edge.W = np.array(self.original.W.T)
        return new_edge
