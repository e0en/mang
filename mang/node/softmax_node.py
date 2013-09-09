import mang.cudamat as cm
from .node import Node
from . import functions as F


class SoftmaxNode(Node):
    def init_training(self, option):
        Node.init_training(self, option)
        self.y_t = cm.empty((self.size, option["batch_size"]))

    def finish_training(self):
        Node.finish_training(self)
        self.y_t.free_device_memory()
        del self.y_t

    def up(self):
        if self.shared:
            raise NotImplementedError

        self._add_b()
        if self.on_gpu:
            if self.y_t.shape != self.y.shape:
                self.y_t.free_device_memory()
                self.y_t = cm.empty((self.y.shape[1], self.y.shape[0]))
            self.y.transpose(self.y_t)
            cm.softmax(self.y_t)
            self.y_t.transpose(self.y)
        else:
            self.y = F.softmax(self.y)

    def down(self):
        pass
        # self.dy.apply_logistic_deriv(self.y)
