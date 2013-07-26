class Node(object):
    def __init__(self, dim, is_output=False, **option):
        self.dim = dim
        self.is_output = is_output

    def f(self, x):
        return x

    def df(self, y):
        return M.ones(y.shape)
