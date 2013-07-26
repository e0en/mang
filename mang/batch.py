import numpy as np
import cudamat.gnumpy as gnp

class Batch:
    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size
        self.N = X[0].shape[0]

        self.idx = np.array(range(self.N))
        np.random.shuffle(self.idx)

    def shuffle(self):
        np.random.shuffle(self.idx)

    def __iter__(self):
        i_batch = 0
        i1 = 0
        while True:
            i2 = i1 + self.batch_size
            if i2 > self.N:
                break

            idx_batch = self.idx[i1:i2]
            yield [x.take(idx_batch, axis=0) for x in self.X]
            i1 += self.batch_size
            if i1 >= self.N:
                break
