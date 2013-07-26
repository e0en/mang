from .node import Node
from . import functions as F


class LogisticNode(Node):
    def f(self, x):
        return F.logistic(x)

    def df(self, y):
        return y*(1. - y)
