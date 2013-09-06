import cudamat as cm
import cudamat.cudamat_conv as cm_conv

from .node import Node
from mang import batch_conv


class ResponseNormalizationNode(Node):
    def __init__(self, dim, option):
        Node.__init__(self, dim, option)
        self.norm_size = option["norm_size"]
        self.add_scale = option["add_scale"]
        self.pow_scale = option["pow_scale"]

    def init_training(self, option):
        Node.init_training(self, option)
        self.x = cm.empty((option["batch_size"], self.size))
        self.cov = cm.empty((option["batch_size"], self.size))
        self.tmp = cm.empty((option["batch_size"], self.size))

    def finish_training(self):
        Node.finish_training(self)
        self.x.free_device_memory()
        self.cov.free_device_memory()
        self.tmp.free_device_memory()
        del self.x, self.cov, self.tmp

    def up(self, x, o=None):
        if o is None:
            x_cm = cm.CUDAMatrix(x)
            o_cm = cm.empty(x.shape)
            tmp_cm = cm.empty(x.shape)
            cm_conv.ResponseNorm(x_cm, tmp_cm, o_cm, self.shape[-1],
                                 self.norm_size, self.add_scale,
                                 self.pow_scale)
            x_cm.free_device_memory()
            tmp_cm.free_device_memory()
            o = o_cm.asarray()
            o_cm.free_device_memory()
            return o
        else:
            cm_conv.ResponseNorm(x, self.cov, o, self.shape[-1],
                                 self.norm_size, self.add_scale,
                                 self.pow_scale)
            self.x.assign(x)

    def down(self, y, do):
        cm_conv.ResponseNormUndo(do, self.cov, y, self.x, self.tmp,
                                 self.shape[-1], self.norm_size, 1., 1.)
        do.assign(self.tmp)
