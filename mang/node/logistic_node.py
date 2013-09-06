from .node import Node
from . import functions as F

import numpy as np
import cudamat as cm


class LogisticNode(Node):
    def up(self, x, o=None):
        (x, o, shape_old) = self._make_shape(x, o)
        if o is None:
            o = self._add_b(x, o)
            o = F.logistic(o)
            return self._recover_shape(x, o, shape_old)
        else:
            self._add_b(x, o)
            o.apply_sigmoid()
            self._recover_shape(x, o, shape_old)

    def down(self, y, dy):
        dy.apply_logistic_deriv(y)
