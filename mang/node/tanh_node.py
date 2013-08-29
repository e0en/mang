import numpy as np
import cudamat.gnumpy as gnp

from .node import Node
from . import functions as F


class TanhNode(Node):
    def up(self, x):
        if isinstance(x, np.ndarray) and self.on_gpu:
            b = self.b.asarray()
        else:
            b = self.b
        return F.tanh(x + b)

    def down(self, y, dy):
        dy *= (1. - y * y)
