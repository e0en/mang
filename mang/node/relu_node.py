import numpy as np
import cudamat.gnumpy as gnp

from .node import Node


class ReLUNode(Node):
    def __init__(self, dim, option):
        Node.__init__(self, dim, option)
        if 'bound' in option:
            self.bound = option["bound"]
        else:
            self.bound = None

    def up(self, x):
        b = self.b_g if isinstance(x, gnp.garray) else self.b
        tmp = x + b

        if self.bound != None:
            tmp = tmp * (tmp <= self.bound) +\
                    self.bound * (tmp > self.bound)
        return tmp * (tmp >= 0.)

    def down(self, y, do):
        if self.bound != None:
            do *= 1. * (y < self.bound) * (y > 0.)
        else:
            do *= 1. * (y > 0.)
