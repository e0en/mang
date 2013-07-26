from .node import Node


class InputNode(Node):
    def f(self, x):
        return x

    def df(self, y):
        return None
