from .node import Node
from . import functions as F


class AffineNode(Node):
    def f(self, x):
        return 1.*x

    def df(self, y):
        return y*0. + 1.
