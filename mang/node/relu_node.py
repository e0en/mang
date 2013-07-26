from mang.node.node import Node
import mang.node.functions as F


class ReLUNode(Node):
    def __init__(self, dim, **option):
        Node.__init__(self, dim, **option)
        if 'bound' in option:
            self.bound = option['bound']
        else:
            self.bound = None

    def f(self, x):
        if self.bound != None:
            tmp = x*(x <= self.bound) + self.bound*(x > self.bound)
            return tmp*(tmp >= 0.)
        else:
            return x*(x >= 0.)

    def df(self, y):
        if self.bound != None:
            return 1.*(y < self.bound)*(y > 0.)
        else:
            return 1.*(y > 0.)
