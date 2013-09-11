from .node import Node
from . import functions as F


class LogisticNode(Node):
    """Node with logistic activation function."""

    _name = "logistic"

    def up(self):
        self._add_b()
        if self.on_gpu:
            self.y.apply_sigmoid()
        else:
            self.y = F.logistic(self.y)

    def down():
        self.dy.apply_logistic_deriv(self.y)
