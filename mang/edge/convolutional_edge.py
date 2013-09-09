import numpy as np

import mang.cudamat as cm
from mang.cudamat import cudamat_conv as cm_conv
from mang.edge.edge import Edge


class ConvolutionalEdge(Edge):
    def __init__(self, shape_in, shape_out, option):
        assert shape_in[0] == shape_in[1]
        assert shape_out[0] == shape_out[1]
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.filter_size = option["filter_size"]
        self.stride = option["stride"]
        if "padding" in option:
            self.padding = option["padding"]
        else:
            self.padding = int(self.filter_size / 2)
        self.n_locs = (shape_in[0] + 2 * self.padding - self.filter_size) /\
            self.stride + 1
        self.n_channel = 1 if len(shape_in) == 2 else shape_in[-1]
        self.n_filter = 1 if len(shape_out) == 2 else shape_out[-1]
        self.shape = \
            (self.n_filter, (self.filter_size ** 2) * self.n_channel)
        self.W = np.zeros(self.shape)
        self.on_gpu = False

    def init_training(self, option):
        Edge.init_training(self, option)
        self.W_tmp = cm.empty(self.W.shape)

    def finish_training(self):
        Edge.finish_training(self)
        self.W_tmp.free_device_memory()
        del self.W_tmp

    def up(self, x, o=None):
        if o is None:
            x_cm = cm.CUDAMatrix(x)
            w_cm = cm.CUDAMatrix(self.W) if not self.on_gpu else self.W
            o_cm = cm.empty((x.shape[0], self.n_filter * self.n_locs ** 2))
            cm_conv.convUp(x_cm, w_cm, o_cm, self.n_locs, self.padding,
                           self.stride, self.n_channel)
            x_cm.free_device_memory()
            if not self.on_gpu:
                w_cm.free_defice_memory()
            o = o_cm.asarray()
            o_cm.free_device_memory()
            return o
        else:
            cm_conv.convUp(x, self.W, o, self.n_locs, self.padding,
                           self.stride, self.n_channel)

    def down(self, do, dx, o, x):
        cm_conv.convDown(do, self.W, dx, self.n_locs, self.padding,
                         self.stride, self.filter_size, self.shape_in[0],
                         self.n_channel)

    def gradient(self, x, do, gW):
        cm_conv.convOutp(x, do, self.W_tmp, self.n_locs, self.padding,
                         self.filter_size, self.stride, self.n_channel)
        # don't divide gradient by # of samples because it is already averaged
        # over a lot of convolution locations
        self.W_tmp.divide(x.shape[0])
        gW.add(self.W_tmp)
