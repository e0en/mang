import Image

import numpy as np
import cudamat.gnumpy as gnp


class Edge(object):
    def __init__(self, node_in, node_out, option):
        self.shape = (node_in.size, node_out.size)
        self.shape_in = node_in.shape
        self.shape_out = node_out.shape

    def init_training(self, option):
        if option["reset"]:
            self.W = option["init_w"] * np.random.randn(*self.shape)
        self.W_g = gnp.garray(self.W)
        self.dW = gnp.zeros(self.W.shape)
        self.gW = gnp.zeros(self.W.shape)

    def copy_params(self):
        self.W = self.W_g.asarray()

    def finish_training(self):
        del self.W_g, self.gW, self.dW

    def up(self, x):
        dot = gnp.dot if isinstance(x, gnp.garray) else np.dot
        W = self.W_g if isinstance(x, gnp.garray) else self.W
        return dot(x, W)

    def down(self, do):
        dot = gnp.dot if isinstance(do, gnp.garray) else np.dot
        W = self.W_g if isinstance(do, gnp.garray) else self.W
        return dot(do, W.T)

    def gradient(self, x, do):
        self.gW += gnp.dot(x.T, do) / x.shape[0]

    def update(self, option):
        momentum = option["momentum"]
        eps = option["eps"] * (1. - momentum)

        self.dW *= momentum
        self.dW += eps * self.gW
        self.W_g += self.dW

        if option["w_norm"] is not None:
            w_norm = (self.W_g * self.W_g).sum(0) + 1e-3
            w_norm = gnp.sqrt(option["w_norm"] / w_norm)
            w_norm = w_norm * (w_norm <= 1.) + (w_norm > 1.)
            self.W_g *= w_norm

    def show(self):
        W = np.array(self.W)
        if len(self.shape_in) == 1:
            img = np.array(W)
            img -= img.min()
            img /= img.max()
            return Image.fromarray(np.uint8(img * 255))

        elif len(self.shape_in) == 2:
            if len(self.shape_out) == 2:
                (w, h) = self.shape_out
            else:
                N = self.shape[1]
                w = int(np.ceil(np.sqrt(self.shape[1]) / 10) * 10)
                h = int(np.ceil(1. * N / w))
            wsize = w * self.shape_in[0]
            hsize = h * self.shape_in[1]
            img = np.zeros((wsize, hsize))

            idx = 0
            for y in xrange(h):
                for x in xrange(w):
                    if idx == N:
                        break
                    tmp = W[:, idx].reshape(self.shape_in)
                    tmp -= tmp.min()
                    tmp /= tmp.max()
                    i1 = x*self.shape_in[0]
                    i2 = i1 + self.shape_in[0]
                    j1 = y*self.shape_in[1]
                    j2 = j1 + self.shape_in[1]
                    img[i1:i2, j1:j2] = tmp
                    idx += 1
            return Image.fromarray(np.uint8(img * 255))
        else:
            raise NotImplementedError

