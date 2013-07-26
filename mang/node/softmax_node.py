from .node import Node
from . import functions as F


class SoftmaxNode(Node):
    def f(self, x):
        return F.softmax(x)

    def df(self, y):
        return y*(1. - y)
