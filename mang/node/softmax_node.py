from mang.node.node import Node
import mang.node.functions as F


class SoftmaxNode(Node):
    def f(self, x):
        return F.softmax(x)

    def df(self, y):
        return y*(1. - y)
