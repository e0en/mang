from .node import Node
from . import functions as F


class TanhNode(Node):
    def f(self, x):
        return F.tanh(x)

    def df(self, y):
        return 1. - y*y
