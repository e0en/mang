import numpy as np
import cudamat.gnumpy as gnp

from .node import Node
from . import functions as F


class SoftmaxNode(Node):
    def up(self, x):
        b = self.b_g if isinstance(x, gnp.garray) else self.b
        return F.softmax(x + b)

    def down(self, y, dy):
        dy *= y * (1. - y)
