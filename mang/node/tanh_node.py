import numpy as np
import cudamat as cm

from .node import Node
from . import functions as F


class TanhNode(Node):
    def up(self, x, o=None):
        (x, o, shape_old) = self._make_shape(x, o)
        if o is None:
            o = self._add_b(x, o)
            o = F.tanh(o)
            return self._recover_shape(x, o, shape_old)
        else:
            self._add_b(x, o)
            cm.tanh(o)
            self._recover_shape(x, o, shape_old)

    def down(self, y, dy):
        dy.apply_tanh_deriv(y)
