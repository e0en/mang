from operator import mul

import numpy as np

import mang.cudamat as cm
from mang.cudamat import cudamat_conv as cm_conv
from mang.edge.edge import Edge


class MaxPoolingEdge(Edge):
    def __init__(self, shape_in, shape_out, option={}):
        self.ratio = option["ratio"]
        self.stride = option["stride"] if "stride" in option else self.ratio
        size_in = reduce(mul, shape_in)
        size_out = reduce(mul, shape_out)
        self.shape = (size_in, size_out)
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.n_channel = 1 if len(shape_in) == 2 else shape_in[-1]
        assert self.n_channel == shape_out[-1]
        self.on_gpu = False
        self.W = np.ones((1, 1))  # dummy weights for consistent interface

    def init_training(self, option):
        Edge.init_training(self, option)
        self.W.assign(1.)
        self.scale = 1.
        self.o = cm.empty((option["batch_size"], self.shape[1]))

    def finish_training(self):
        Edge.finish_training(self)
        self.o.free_device_memory()
        del self.o

    def up(self, x, o):
        if self.o.shape != o.shape:
            self.o.free_device_memory()
            self.o = cm.empty((x.shape[0], self.shape[1]))
        cm_conv.MaxPool(x, self.o, self.n_channel, self.ratio, 0,
                        self.stride, self.shape_out[0])
        o.assign(self.o)
        o.mult(self.scale)

    def down(self, do, dx, o, x):
        cm_conv.MaxPoolUndo(x, dx, do, self.o, self.ratio, 0, self.stride,
                            self.shape_out[0])
        dx.mult(self.scale)

    def gradient(self, x, do, gW):
        gW.assign(0)
        self.scale = float(self.W.asarray()[0])
