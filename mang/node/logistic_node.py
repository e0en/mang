from .node import Node
from . import functions as F

import numpy as np
import cudamat.gnumpy as gnp


class LogisticNode(Node):
    def up(self, x):
        if isinstance(x, np.ndarray) and self.on_gpu:
            b = self.b.asarray()
        else:
            b = self.b
        return F.logistic(x + b)

    def down(self, y, dy):
        dy *= y * (1. - y)
