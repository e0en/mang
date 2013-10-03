from .node import Node


class LogisticNode(Node):
    """Node with logistic activation function."""

    _name = "logistic"

    def up(self):
        self._add_b()
        self.y.apply_sigmoid()

    def down(self):
        self.dy.apply_logistic_deriv(self.y)
