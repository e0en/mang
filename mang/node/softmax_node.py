import cudamat as cm

from .node import Node
from . import functions as F


class SoftmaxNode(Node):
    def init_training(self, option):
        Node.init_training(self, option)
        self.o_t = cm.empty((self.size, option["batch_size"]))

    def finish_training(self):
        self.o_t.free_device_memory()

    def up(self, x, o=None):
        if self.shared:
            raise NotImplementedError

        if o is None:
            if self.use_bias:
                b = self.b.asarray() if self.on_gpu else self.b
                o = F.softmax(x + b)
            else:
                o = F.softmax(x)
            return o
        else:
            if self.use_bias:
                x.add_row_vec(self.b, o)
            else:
                o.assign(x)
            o.transpose(self.o_t)
            cm.softmax(self.o_t)
            self.o_t.transpose(o)

    def down(self, y, dy):
        dy.apply_logistic_deriv(y)
