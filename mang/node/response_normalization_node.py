import mang.cudamat as cm
from mang.cudamat import cudamat_conv as cm_conv

from .node import Node


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

    def up(self):
        if self.on_gpu:
            if self.x.shape != self.y.shape:
                self.x.free_device_memory()
                self.cov.free_device_memory()
                self.tmp.free_device_memory()
                self.x = cm.empty(self.y.shape)
                self.cov = cm.empty(self.y.shape)
                self.tmp = cm.empty(self.y.shape)
            self.x.assign(self.y)
            cm_conv.ResponseNorm(self.y, self.cov, self.y,
                                 self.shape[-1],
                                 self.norm_size, self.add_scale,
                                 self.pow_scale)
        else:
            raise NotImplementedError

    def down(self):
        cm_conv.ResponseNormUndo(self.dy, self.cov, self.y, self.x,
                                 self.tmp, self.shape[-1],
                                 self.norm_size, 1., 1.)
        self.dy.assign(self.tmp)
