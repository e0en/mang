from .node import Node
from . import functions as F

import numpy as np
import cudamat.gnumpy as gnp


class LogisticNode(Node):
    def up(self, x):
        b = self.b_g if isinstance(x, gnp.garray) else self.b
        return F.logistic(x + b)

    def down(self, y, dy):
        dy *= y * (1. - y)
