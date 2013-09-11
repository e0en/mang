import numpy as np

import mang.cudamat as cm
from mang.cudamat import cudamat_conv as cm_conv
from mang.edge.edge import Edge


class ConvolutionalEdge(Edge):

    _name = "conv"

    def __init__(self, nodes, conn, option):
        self.conn = conn
        (node1, node2) = (nodes[conn[0]], nodes[conn[1]])
        assert node1.shape[0] == node1.shape[1]
        assert node2.shape[0] == node2.shape[1]

        # user-defined properties
        self.filter_size = option["filter_size"]
        self.stride = option["stride"] if "stride" in option else 1
        if "padding" in option:
            self.padding = option["padding"]
        else:
            self.padding = int(self.filter_size / 2)

        # derived property
        self.n_locs = (node1.shape[0] + 2 * self.padding - self.filter_size) /\
            self.stride + 1
        self.n_channel = 1 if len(node1.shape) == 2 else node1.shape[-1]
        self.n_filter = 1 if len(node2.shape) == 2 else node2.shape[-1]
        self.shape = \
            (self.n_filter, (self.filter_size ** 2) * self.n_channel)

        if "W" in option:
            self.W = np.array(option["W"], dtype=np.float32, order="F")
        else:
            self.W = np.zeros(self.shape)

        self.on_gpu = False
        self.used_gpu_memory = 0

    def init_training(self, batch_size):
        Edge.init_training(self, batch_size)
        self.W_tmp = cm.empty(self.W.shape)
        self.used_gpu_memory += 4 * self.W.shape[0] * self.W.shape[1]

    def finish_training(self):
        Edge.finish_training(self)
        self.W_tmp.free_device_memory()
        self.used_gpu_memory -= 4 * self.W.shape[0] * self.W.shape[1]
        del self.W_tmp

    def up(self, nodes):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        cm_conv.convUp(node1.y, self.W, node2.y, self.n_locs, self.padding,
                       self.stride, self.n_channel)

    def down(self, nodes):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        cm_conv.convDown(node2.dy, self.W, node1.dy, self.n_locs, self.padding,
                         self.stride, self.filter_size, node1.shape[0],
                         self.n_channel)

    def gradient(self, nodes, gW):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        cm_conv.convOutp(
            node1.y, node2.dy, self.W_tmp, self.n_locs, self.padding,
            self.filter_size, self.stride, self.n_channel)
        # don't divide gradient by # of samples because it is already averaged
        # over a lot of convolution locations
        self.W_tmp.divide(node1.y.shape[0])
        gW.add(self.W_tmp)

    def to_dict(self):
        """Convert self to a dict."""

        result = Edge.to_dict(self)
        result["filter_size"] = self.filter_size
        result["padding"] = self.padding
        result["stride"] = self.stride
        return result
