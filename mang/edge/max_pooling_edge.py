from operator import mul

import numpy as np

import mang.cudamat as cm
from mang.cudamat import cudamat_conv as cm_conv
from mang.edge.edge import Edge


class MaxPoolingEdge(Edge):

    _name = "max_pool"

    def __init__(self, nodes, conn, option={}):
        self.conn = conn
        (node1, node2) = (nodes[conn[0]], nodes[conn[1]])

        self.ratio = option["ratio"]
        self.stride = option["stride"] if "stride" in option else self.ratio

        size_in = reduce(mul, node1.shape)
        size_out = reduce(mul, node2.shape)
        self.shape = (size_in, size_out)
        self.n_channel = 1 if len(node1.shape) == 2 else node1.shape[-1]
        assert self.n_channel == node2.shape[-1]

        self.on_gpu = False
        self.W = np.ones((1, 1))  # dummy weights for consistent interface
        self.o = None
        self.dy = None
        self.used_gpu_memory = 0

    def to_gpu(self, batch_size):
        if not self.on_gpu:
            self.o = cm.empty((batch_size, self.shape[1]))
            self.used_gpu_memory += self.o.shape[0] * self.o.shape[1] * 4
            self.dy = cm.empty((batch_size, self.shape[0]))
            self.used_gpu_memory += self.dy.shape[0] * self.dy.shape[1] * 4
        Edge.to_gpu(self, batch_size)

    def from_gpu(self):
        if self.on_gpu:
            self.o.free_device_memory()
            self.used_gpu_memory -= self.o.shape[0] * self.o.shape[1] * 4
            self.dy.free_device_memory()
            self.used_gpu_memory -= self.dy.shape[0] * self.dy.shape[1] * 4
            del self.o, self.dy
        Edge.from_gpu(self)

    def init_training(self, batch_size):
        self.to_gpu(batch_size)
        self.W.assign(1.)

    def finish_training(self):
        self.from_gpu()

    def up(self, nodes):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        cm_conv.MaxPool(node1.y, self.o, self.n_channel, self.ratio, 0,
                        self.stride, node2.shape[0])
        node2.y.assign(self.o)

    def down(self, nodes):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        cm_conv.MaxPoolUndo(
            node1.y, self.dy, node2.dy, self.o, self.ratio, 0, self.stride,
            node2.shape[0])
        node1.dy.add(self.dy)

    def gradient(self, nodes, gW):
        gW.assign(0)

    def to_dict(self):
        """Convert self to a dict."""

        result = Edge.to_dict(self)
        result["ratio"] = self.ratio
        result["stride"] = self.stride
        return result
