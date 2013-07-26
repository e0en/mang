from mang.node.node import Node
import mang.node.functions as F


class AffineNode(Node):
    def f(self, x):
        return 1.*x

    def df(self, y):
        return y*0. + 1.
