from operator import mul

import numpy as np

import mang.cudamat as cm


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

    def to_gpu(self, n_replica):
        """Initialize GPU variables."""

        if not self.on_gpu:
            self.b = cm.CUDAMatrix(self.b.reshape(1, self.b.size))
            self.y = cm.empty((n_replica, self.size))
            self.n_replica = n_replica
            self.on_gpu = True
        elif n_replica != self.n_replica:
            self.from_gpu()
            self.to_gpu(n_replica)

    def from_gpu(self):
        """Copy GPU variables to CPU and free allocated GPU memory."""

        if self.on_gpu:
            b = self.b.asarray()
            self.b.free_device_memory()
            self.b = b.reshape((b.size, ))
            self.y.free_device_memory()
            del self.n_replica
            self.on_gpu = False

    def init_training(self, option):
        """Create temporary GPU variables."""

        self.to_gpu(option["batch_size"])
        self.dy = cm.empty((self.n_replica, self.size))

    def finish_training(self):
        """Copy GPU variables to host, and free allocated GPU memory."""

        self.from_gpu()
        self.dy.free_device_memory()
        del self.dy

    def _make_shape(self):
        """Reshape matrix variables to add shared bias without problem."""

        if self.shared:
            n_elem = self.n_replica * self.size
            shape_new = (n_elem / self.shape[-1], self.shape[-1])
            if self.on_gpu:
                self.y = self.y.reshape(shape_new)
            else:
                self.y.reshape(shape_new)

    def _recover_shape(self):
        """Recover the shape of matrix variables reshaped by _make_shape."""
        if self.shared:
            shape_old = (self.n_replica, self.size)
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
            gb.divide(self.n_replica)

            if self.shared:
                self.dy.reshape(shape_old)
