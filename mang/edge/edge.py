import numpy as np
import cudamat.gnumpy as gnp


class Edge(object):
    def __init__(self, node_in, node_out, option):
        self.shape = (node_in.size, node_out.size)
        self.shape_in = node_in.shape
        self.shape_out = node_out.shape
        self.W = np.zeros(self.shape)
        self.on_gpu = False

    def init_training(self):
        self.W = gnp.garray(self.W)
        self.on_gpu = True

    def finish_training(self):
        self.W = self.W.asarray()
        self.on_gpu = False

    def up(self, x):
        dot = gnp.dot if isinstance(x, gnp.garray) else np.dot
        if isinstance(x, np.ndarray) and self.on_gpu:
            W = self.W.asarray()
        else:
            W = self.W
        return dot(x, W)

    def down(self, do):
        return gnp.dot(do, self.W.T)

    def gradient(self, x, do):
        return gnp.dot(x.T, do) / x.shape[0]
