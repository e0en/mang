import mang.cudamat as cm
from .node import Node
from . import functions as F


class TanhNode(Node):
    "Node with hyperbolic tangent activation function."""

    _name = "tanh"

    def up(self):
        self._add_b()

        if self.on_gpu:
            cm.tanh(self.y)
        else:
            self.y = F.tanh(self.y)

    def down(self):
        self.dy.apply_tanh_deriv(self.y)
