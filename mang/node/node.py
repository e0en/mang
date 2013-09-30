from operator import mul

import numpy as np

import mang.cudamat as cm


class Node(object):
    """Node with linear(or identity) activation function."""
    _name = "affine"

    def __init__(self, shape, option):
        self.shape = shape
        self.size = reduce(mul, self.shape)

        if "use_bias" in option:
            self.use_bias = option["use_bias"]
        else:
            self.use_bias = True

        if self.use_bias:
            if "shared" in option and option["shared"]:
                self.b = np.zeros((self.shape[-1], ))
                self.shared = True
            else:
                self.b = np.zeros((self.size, ))
                self.shared = False
        else:
            self.b = None
            self.shared = False

        if "b" in option and self.use_bias:
            self.b = np.array(option["b"], dtype=np.float32, order="F")

        self.on_gpu = False
        self.used_gpu_memory = 0

    def to_gpu(self, batch_size):
        """Initialize GPU variables."""

        if not self.on_gpu:
            if self.use_bias:
                self.b = cm.CUDAMatrix(self.b.reshape(1, self.b.size))
                self.used_gpu_memory += 4 * self.b.shape[0] * self.b.shape[1]
            self.y = cm.empty((batch_size, self.size))
            self.used_gpu_memory += 4 * self.y.shape[0] * self.y.shape[1]

            self.batch_size = batch_size
            self.on_gpu = True

    def from_gpu(self):
        """Copy GPU variables to CPU and free allocated GPU memory."""

        if self.on_gpu:
            if self.use_bias:
                b = self.b.asarray()
                self.b.free_device_memory()
                self.b = b.reshape((b.size, ))
                self.used_gpu_memory -= 4 * self.b.size
            self.y.free_device_memory()
            self.used_gpu_memory -= self.y.shape[0] * self.y.shape[1] * 4

            del self.batch_size
            self.on_gpu = False

    def init_training(self, option):
        """Create temporary GPU variables."""

        self.to_gpu(option["batch_size"])
        self.dy = cm.empty((self.batch_size, self.size))
        self.used_gpu_memory += self.dy.shape[0] * self.dy.shape[1] * 4

    def finish_training(self):
        """Copy GPU variables to host, and free allocated GPU memory."""

        self.from_gpu()
        self.dy.free_device_memory()
        self.used_gpu_memory -= self.dy.shape[0] * self.dy.shape[1] * 4
        del self.dy

    def _make_shape(self):
        """Reshape matrix variables to add shared bias without problem."""

        if self.shared:
            n_elem = self.batch_size * self.size
            shape_new = (n_elem / self.shape[-1], self.shape[-1])
            if self.on_gpu:
                self.y = self.y.reshape(shape_new)
            else:
                self.y.reshape(shape_new)

    def _recover_shape(self):
        """Recover the shape of matrix variables reshaped by _make_shape."""
        if self.shared:
            shape_old = (self.batch_size, self.size)
            if self.on_gpu:
                self.y.reshape(shape_old)
            else:
                self.y = self.y.reshape(shape_old)

    def _add_b(self):
        """Add bias to input signal."""

        if self.use_bias:
            self._make_shape()
            if self.on_gpu:
                self.y.add_row_vec(self.b)
            else:
                self.y += self.b
            self._recover_shape()

    def up(self):
        """Calculate activation from input signals."""

        self._add_b()

    def down(self):
        """Calculate error to back-propagate from input error signals."""

        pass  # do nothing

    def gradient(self, gb):
        """Calculate the gradient of the bias."""

        if self.use_bias:
            if self.shared:
                shape_old = self.dy.shape
                n_elem = self.dy.shape[0] * self.dy.shape[1]
                shape_new = (n_elem / self.shape[-1], self.shape[-1])
                self.dy.reshape(shape_new)

            self.dy.sum(0, gb)  # should be changed to the sum over axis 1
            gb.divide(self.batch_size)

            if self.shared:
                self.dy.reshape(shape_old)

    def to_dict(self):
        """Convert self into a dictionary."""

        result = {
            "b": self.b,
            "shape": self.shape,
            "shared": self.shared,
            "use_bias": self.use_bias,
            "type": self._name,
            }
        return result

    @classmethod
    def from_dict(cls, data):
        """Create a node object from a dictionary."""

        return cls(data["shape"], data)
