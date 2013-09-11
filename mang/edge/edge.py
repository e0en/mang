from operator import mul

import numpy as np

import mang.cudamat as cm


class Edge(object):

    _name = "full"

    def __init__(self, nodes, conn, option={}):
        self.conn = conn
        size_in = reduce(mul, nodes[conn[0]].shape)
        size_out = reduce(mul, nodes[conn[1]].shape)
        self.shape = (size_in, size_out)
        self.W = option["W"] if "W" in option else np.zeros(self.shape)
        self.on_gpu = False
        self.used_gpu_memory = 0

    def to_gpu(self, batch_size):
        if not self.on_gpu:
            self.W = cm.CUDAMatrix(self.W)
            self.used_gpu_memory += self.W.shape[0] * self.W.shape[1] * 4
            self.on_gpu = True

    def from_gpu(self):
        if self.on_gpu:
            W = self.W.asarray()
            self.W.free_device_memory()
            self.used_gpu_memory -= self.W.shape[0] * self.W.shape[1] * 4
            self.W = W
            self.on_gpu = False

    def init_training(self, option):
        self.to_gpu(option)

    def finish_training(self):
        self.from_gpu()

    def up(self, nodes):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        node2.y.add_dot(node1.y, self.W)

    def down(self, nodes):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        node1.dy.add_dot(node2.dy, self.W.T)

    def gradient(self, nodes, gW):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        gW.add_dot(node1.y.T, node2.dy, 1. / node1.y.shape[0])

    def to_dict(self):
        """Convert self into a dict."""

        W = self.W.asarray() if self.on_gpu else self.W
        result = {
            "type": self._name,
            "conn": self.conn,
            "W": W,
            }
        return result

    @classmethod
    def from_dict(cls, nodes, data):
        """Create an edge object from a dict."""

        return cls(nodes, data["conn"], data)
