import numpy as np
import cudamat.gnumpy as gnp

from .node import Node


class ReLUNode(Node):
    def __init__(self, dim, option):
        Node.__init__(self, dim, option)

    def up(self, x):
        if isinstance(x, np.ndarray) and self.on_gpu:
            b = self.b.asarray()
        else:
            b = self.b
        tmp = x + b
        return tmp * (tmp >= 0.)

    def down(self, y, do):
        do *= 1. * (y > 0.)
