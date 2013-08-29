import numpy as np
import cudamat.gnumpy as gnp


class EdgeRef(object):
    def __init__(self, original, option):
        self.original = original
        self.transpose = option["transpose"]

    def init_training(self):
        pass

    def finish_training(self):
        pass

    def up(self, x):
        dot = gnp.dot if isinstance(x, gnp.garray) else np.dot
        if isinstance(x, np.ndarray) and self.original.on_gpu:
            W = self.original.W.asarray()
        else:
            W = self.original.W

        if self.transpose:
            return dot(x, W.T)
        else:
            return dot(x, W)

    def down(self, do):
        if self.transpose:
            return gnp.dot(do, self.original.W)
        else:
            return gnp.dot(do, self.original.W.T)

    def gradient(self, x, do):
        if self.transpose:
            return gnp.dot(do.T, x) / x.shape[0]
        else:
            return gnp.dot(x.T, do) / x.shape[0]
