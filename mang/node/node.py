from operator import mul

import numpy as np
import cudamat as cm


class Node(object):
    """Node with linear(or identity) activation function."""
    def __init__(self, shape, option):
        self.shape = shape
        self.size = reduce(mul, self.shape)
        if "use_bias" in option:
            self.use_bias = option["use_bias"]
        else:
            self.use_bias = True
        if "shared" in option and option["shared"]:
            self.b = np.zeros((self.shape[-1], ))
            self.shared = True
        else:
            self.b = np.zeros(self.size)
            self.shared = False
        self.on_gpu = False

    def init_training(self, option):
        """Initialize temporary GPU variables."""
        self.b = cm.CUDAMatrix(self.b.reshape(1, self.b.size))
        self.on_gpu = True

    def finish_training(self):
        b = self.b.asarray()
        self.b.free_device_memory()
        self.b = b.reshape((b.size, ))
        self.on_gpu = False

    def _make_shape(self, x, o):
        if self.shared:
            shape_old = x.shape
            n = x.shape[0] * x.shape[1]
            shape_new = (n / self.shape[-1], self.shape[-1])
            if o is None:
                x = x.reshape(shape_new)
            else:
                x.reshape(shape_new)
                o.reshape(shape_new)
            return (x, o, shape_old)
        else:
            return (x, o, None)

    def _recover_shape(self, x, o, shape_old):
        if self.shared:
            if not isinstance(x, cm.CUDAMatrix):
                x = x.reshape(shape_old)
                o = o.reshape(shape_old)
                return o
            else:
                x.reshape(shape_old)
                o.reshape(shape_old)
        elif not isinstance(x, cm.CUDAMatrix):
            return o


    def _add_b(self, x, o):
        if self.use_bias:
            if o is None:
                b = self.b.asarray() if self.on_gpu else self.b
                return x + b
            else:
                x.add_row_vec(self.b, o)
        elif o is None:
            return x

    def up(self, x, o=None):
        (x, o, shape_old) = self._make_shape(x, o)
        if o is None:
            o = self._add_b(x, o)
            return self._recover_shape(x, o, shape_old)
        else:
            x.add_row_vec(self.b, o)
            self._recover_shape(x, o, shape_old)

    def down(self, y, dy):
        pass  # do nothing

    def gradient(self, dy, gb):
        if not self.use_bias:
            return

        n = dy.shape[0]
        if self.shared:
            shape_old = dy.shape
            n = dy.shape[0] * dy.shape[1]
            shape_new = (n / self.shape[-1], self.shape[-1])
            dy.reshape(shape_new)

        dy.sum(0, gb)
        gb.divide(n)

        if self.shared:
            dy.reshape(shape_old)
