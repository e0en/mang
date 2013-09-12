import mang.cudamat as cm
from .node import Node


class TanhNode(Node):
    "Node with hyperbolic tangent activation function."""

    _name = "tanh"

    def up(self):
        self._add_b()
        cm.tanh(self.y)

    def down(self):
        self.dy.apply_tanh_deriv(self.y)
