import mang.cudamat as cm
from mang.cudamat import cudamat_conv as cm_conv

from .node import Node


class ResponseNormalizationNode(Node):
    """Node that performs response normalization."""

    _name = "rnorm"

    def __init__(self, shape, option):
        option["use_bias"] = False
        Node.__init__(self, shape, option)
        self.norm_size = option["norm_size"]
        self.add_scale = option["add_scale"]
        self.pow_scale = option["pow_scale"]

        self.x = None
        self.cov = None
        self.tmp = None

    def _make_tmp(self):
        self.x = cm.empty(self.y.shape)
        self.cov = cm.empty(self.y.shape)
        self.tmp = cm.empty(self.y.shape)
        self.used_gpu_memory += 12 * self.x.shape[0] * self.x.shape[1]

    def _free_tmp(self):
        self.x.free_device_memory()
        self.cov.free_device_memory()
        self.tmp.free_device_memory()
        self.used_gpu_memory -= 12 * self.x.shape[0] * self.x.shape[1]
        self.x = None
        self.cov = None
        self.tmp = None

    def init_training(self, option):
        Node.init_training(self, option)
        self._make_tmp()

    def finish_training(self):
        Node.finish_training(self)
        self._free_tmp()

    def up(self):
        if self.x is None:
            self._make_tmp()
        self.x.assign(self.y)
        cm_conv.ResponseNorm(
            self.y, self.cov, self.y, self.shape[-1], self.norm_size,
            self.add_scale, self.pow_scale)

    def down(self):
        cm_conv.ResponseNormUndo(self.dy, self.cov, self.y, self.x,
                                 self.tmp, self.shape[-1],
                                 self.norm_size, 1., 1.)
        self.dy.assign(self.tmp)

    def to_dict(self):
        """Convert self into a dictionary."""

        result = Node.to_dict(self)
        result["add_scale"] = self.add_scale
        result["pow_scale"] = self.pow_scale
        result["norm_size"] = self.norm_size

        return result

    @staticmethod
    def from_dict(data):
        """Create a node object from a dictionary."""

        return ResponseNormalizationNode(data["shape"], data)
