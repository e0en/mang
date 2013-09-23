from .node import Node


class ReLUNode(Node):
    """Node with rectified linear activation function."""
    _name = "relu"

    def __init__(self, dim, option):
        Node.__init__(self, dim, option)

    def up(self):
        self._add_b()
        self.y.lower_bound(0.)

    def down(self):
        self.dy.apply_rectified_linear_deriv(self.y)
