from mang.node.node import Node
import mang.node.functions as F


class InvCubicNode(Node):
    def __init__(self, dim, **option):
        Node.__init__(self, dim, **option)

    def f(self, x):
        return F.inv_cubic(x)

    def df(self, y):
        return 1./(y**2 + 1)
