from .node import Node


class ReLUNode(Node):
    def __init__(self, dim, option):
        Node.__init__(self, dim, option)

    def up(self):
        self._add_b()
        if self.on_gpu:
            self.y.lower_bound(0.)
        else:
            self.y = self.y * (self.y >= 0.)

    def down(self):
        self.dy.apply_rectified_linear_deriv(self.y)
