from mang.edge import Edge


class EdgeRef(object):

    _name = "ref"

    def __init__(self, conn, original, option):
        self.conn = conn
        self.original = original
        if 'transpose' in option:
            self.transpose = option["transpose"]
        else:
            self.transpose = False

    def init_training(self, option):
        pass

    def finish_training(self):
        pass

    def up(self, nodes):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        if self.transpose:
            node2.y.add_dot(node1.y, self.original.W.T)
        else:
            node2.y.add_dot(node1.y, self.original.W)

    def down(self, nodes):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        if self.transpose:
            node1.dy.add_dot(node2.dy, self.original.W)
        else:
            node1.dy.add_dot(node2.dy, self.original.W.T)

    def gradient(self, nodes, gW):
        (node1, node2) = (nodes[self.conn[0]], nodes[self.conn[1]])
        if self.transpose:
            gW.add_dot(node2.dy.T, node1.y, 1. / node1.y.shape[0])
        else:
            gW.add_dot(node1.y.T, node2.dy, 1. / node1.y.shape[0])

    def materialize(self, nodes):
        W = self.original.W if self.transpose else self.original.W.T
        return Edge(nodes, self.conn, W)

    def to_dict(self):
        """Convert self to a dict."""

        result = {"transpose": self.transpose, "conn": self.conn, }
        return result

    @classmethod
    def from_dict(cls, original, data):
        """Create an edge reference object from data."""

        return cls(data["conn"], original, data)
