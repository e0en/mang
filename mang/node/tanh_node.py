import numpy as np
import cudamat.gnumpy as gnp

from .node import Node
from . import functions as F


class TanhNode(Node):
    def up(self, x):
        b = self.b_g if isinstance(x, gnp.garray) else self.b
        return F.tanh(x + b)

    def down(self, y, dy):
        dy *= (1. - y * y)
