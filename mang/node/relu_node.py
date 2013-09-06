from .node import Node


class ReLUNode(Node):
    def __init__(self, dim, option):
        Node.__init__(self, dim, option)

    def up(self, x, o=None):
        (x, o, shape_old) = self._make_shape(x, o)
        if o is None:
            o = self._add_b(x, o)
            o = o * (o >= 0.)
            return self._recover_shape(x, o, shape_old)
        else:
            self._add_b(x, o)
            o.lower_bound(0.)
            self._recover_shape(x, o, shape_old)

    def down(self, y, do):
        do.apply_rectified_linear_deriv(y)
