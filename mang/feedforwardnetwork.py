import numpy as np
import cudamat.gnumpy as gnp

from mang.measure import measure_table
from mang.cost import d_cost_table
from mang.noise import noise_table
from mang import graph
from mang import node as mnode
from mang import edge as medge


_DEFAULT_OPTION = {
    "reset": True,
    "init_w": 0.01,
    "n_epoch": 1000,
    "batch_size": 100,
    "chunk_size": 10000,
    "dropout": 0.5,
    "drop_view": True,
    "display": True,
    "momentum": 0.99,
    "eps": .1,
    "w_norm": 15.,
    "noise": None,
    "noise_param": None,
    "measure": None,
    "cost": None,
    "cost_scale": None,
    "callback": None,
    }


_EPS = 1e-6


class FeedForwardNetwork(object):
    def __init__(self, nodes, edges):
        self.nodes = {}
        for name in nodes:
            spec = nodes[name]
            node_type = mnode.node_table[spec["type"]]
            self.nodes[name] = node_type(spec["shape"], spec)
        self.n_node = len(self.nodes.keys())

        self.edges = {}
        self.ref_count = {}
        for conn in edges:
            spec = edges[conn]
            if spec["type"] != "ref":
                edge_type = medge.edge_table[spec["type"]]
                node_in = self.nodes[conn[0]]
                node_out = self.nodes[conn[1]]
                self.edges[conn] = edge_type(node_in, node_out, spec)
                self.ref_count[conn] = 1.

        ref_edges = [x for x in edges if edges[x]["type"] == "ref"]
        self.ref_graph = {}
        for conn in ref_edges:
            spec = edges[conn]
            if "option" not in spec:
                spec["option"] = {}
            self.ref_graph[conn] = spec["original"]
            self.ref_count[spec["original"]] += 1
            edge_type = medge.edge_table["ref"]
            edge_original = self.edges[spec["original"]]
            self.edges[conn] = edge_type(edge_original, spec)

        (self.inputs, self.outputs) = graph.find_boundary(self.edges.keys())
        self.boundary = self.inputs | self.outputs
        self.n_input = len(self.inputs)

        # ff_edges: outgoing edges, bp_edges: incoming edges
        self.ff_edges = dict(
            [(name, [x[1] for x in self.edges if x[0] == name])
                for name in self.nodes])
        self.bp_edges = dict(
            [(name, [x[0] for x in self.edges if x[1] == name])
                for name in self.nodes])

        self.ff_order = graph.find_order(self.edges.keys())
        self.bp_order = list(self.ff_order)
        self.bp_order.reverse()

        self.is_training = False

    def _init_training(self, option):
        # load default options
        _option = dict(_DEFAULT_OPTION)
        for key in option:
            _option[key] = option[key]
        option = _option

        if len(self.inputs) == 1:
            option["drop_view"] = False

        for conn in self.edges:
            if conn not in self.ref_graph:
                self.edges[conn].W *= 0
                self.edges[conn].W += option["init_w"] * \
                    np.random.randn(*self.edges[conn].shape)
                self.edges[conn].init_training()
        for name in self.nodes:
            self.nodes[name].b *= 0
            self.nodes[name].init_training()

        if option["cost"] is None:
            # squared error is the default cost function
            for name in self.outputs:
                if name not in option["cost"]:
                    option["cost"][name] = d_cost_table["squared_error"]

        if option["measure"] is None:
            # RMSE is the default performance measure
            option["measure"] = {}
            for name in self.outputs:
                if name not in option["measure"]:
                    option["measure"][name] = measure_table["rmse"]

        # temporary variables used for training
        # activations, and their gradients
        self._g = {"data": {}, "y": {}, "do": {}, "db": {}, "gb": {},
                   "dW": {}, "gW": {}, }

        # allocate spaces for data and biases
        batch_size = option["batch_size"]
        for name in self.boundary:
            self._g["data"][name] = \
                gnp.zeros((batch_size, self.nodes[name].size))
        for name in self.nodes:
            self._g["y"][name] = \
                gnp.zeros((batch_size, self.nodes[name].size))
            self._g["do"][name] = \
                gnp.zeros((batch_size, self.nodes[name].size))

        for name in self.nodes:
            self._g["db"][name] = gnp.zeros((self.nodes[name].size, ))
            self._g["gb"][name] = gnp.zeros((self.nodes[name].size, ))

        for conn in self.edges:
            if conn not in self.ref_graph:
                self._g["dW"][conn] = gnp.zeros(self.edges[conn].shape)
                self._g["gW"][conn] = gnp.zeros(self.edges[conn].shape)

        # allocate spaces for dropout masks (still not sure if it's needed)
        if option["dropout"] > 0.:
            self._g["mask"] = {}
            for name in self.nodes:
                self._g["mask"][name] = \
                    gnp.zeros((batch_size, self.nodes[name].size))

        if option["drop_view"]:
            self._g["drop_mask"] = gnp.zeros((self.n_input + 1, batch_size))

        self.train_option = dict(option)
        self.is_training = True

    def _finish_training(self):
        """store the trained parameters and remove GPU variables"""
        for conn in self.edges:
            if conn not in self.ref_graph:
                self.edges[conn].finish_training()
        for name in self.nodes:
            self.nodes[name].finish_training()
            if self.train_option["dropout"] > 0.:
                if name not in self.inputs:
                    for name2 in self.ff_edges[name]:
                        conn = (name, name2)
                        self.edges[conn].W *= self.train_option["dropout"]

        del self._g, self.train_option
        gnp.free_reuse_cache()
        reload(gnp)
        self.is_training = False

    def fit(self, data, **option):
        self._init_training(option)
        momentum_f = self.train_option["momentum"]
        for i_epoch in xrange(option["n_epoch"]):
            option["eps"] *= 0.998
            if i_epoch < 500:
                r_epoch = i_epoch / 500.
                self.train_option["momentum"] = r_epoch * momentum_f + \
                    0.5 * (1. - r_epoch)
            else:
                self.train_option["momentum"] = momentum_f
            self.fit_epoch(data)
            if self.train_option["callback"] is not None:
                self.train_option["callback"](self, i_epoch)
        self._finish_training()

    def fit_epoch(self, data):
        N = data[data.keys()[0]].shape[0]
        batch_size = self.train_option["batch_size"]
        assert set(self.boundary) == set(data.keys())

        # shuffle data
        rng_state = np.random.get_state()
        for name in data:
            np.random.set_state(rng_state)
            np.random.shuffle(data[name])

        n_batch = int(N / batch_size)
        i1 = 0
        for i_batch in xrange(n_batch):
            i2 = i1 + batch_size
            for name in data:
                self._g["data"][name] = gnp.garray(data[name][i1:i2])

            if self.train_option["noise"]:
                # add pre-specified noise pattern to input samples
                for name in self.inputs:
                    noise_func = noise_table[self.train_option["noise"][name]]
                    param = self.train_option["noise_param"][name]
                    self._g["data"][name] = \
                        noise_func(self._g["data"][name], param)
            if self.train_option["drop_view"]:
                # randomly drop views if user chose to do so
                self._g["drop_mask"] = gnp.rand(self.n_input + 1, batch_size)
                prob = 1. / (self.n_input + 1)
                for (i, name) in enumerate(self.inputs):
                    tmp = self._g["drop_mask"][self.n_input] <= prob
                    tmp += self._g["drop_mask"][i] <= prob
                    self._g["data"][name] = (self._g["data"][name].T * tmp).T

            # mini-batch training
            self.fit_step()

            i1 = i2

    # calculate the gradient using BP and update parameters
    def fit_step(self):
        self._feed_forward()
        self._back_propagate()
        self._update()

    def _feed_forward(self):
        for name in self.nodes:
            if name in self.inputs:
                self._g["y"][name] = self._g["data"][name]
            else:
                self._g["y"][name] *= 0
        for name in self.ff_order:
            if name not in self.inputs:
                self._g["y"][name] = self.nodes[name].up(self._g["y"][name])
                # dropout: randomly drop hidden nodes using binary masks
                dropout = self.train_option["dropout"]
                if dropout > 0. and name not in self.outputs:
                    self._g["mask"][name] = \
                        gnp.rand(*self._g["y"][name].shape) > dropout
                    self._g["y"][name] *= self._g["mask"][name]

            for name2 in self.ff_edges[name]:
                conn = (name, name2)
                self._g["y"][name2] += self.edges[conn].up(self._g["y"][name])

    def _back_propagate(self):
        for name in self.nodes:
            if name not in self.boundary:
                self._g["do"][name] *= 0
            elif name in self.outputs:
                cost_func = d_cost_table[self.train_option["cost"][name]]
                self._g["do"][name] = \
                    cost_func(self._g["y"][name], self._g["data"][name])
        for conn in self.edges:
            if conn not in self.ref_graph:
                self._g["gW"][conn] *= 0

        for name in self.bp_order:
            if name not in self.inputs:
                self.nodes[name].down(self._g["y"][name], self._g["do"][name])
                self._g["gb"][name] = \
                    self.nodes[name].gradient(self._g["do"][name])
                # dropout: randomly drop hidden nodes using binary masks
                dropout = self.train_option["dropout"]
                if dropout > 0. and name not in self.outputs:
                    self._g["do"][name] *= self._g["mask"][name]
            for name2 in self.bp_edges[name]:
                conn = (name2, name)
                if conn in self.ref_graph:
                    conn2 = self.ref_graph[conn]
                else:
                    conn2 = conn
                self._g["gW"][conn2] += self.edges[conn].gradient(
                    self._g["y"][name2], self._g["do"][name])
                if name2 not in self.inputs:
                    self._g["do"][name2] += \
                        self.edges[conn].down(self._g["do"][name])

    def _update(self):
        momentum = self.train_option["momentum"]
        eps = (1. - momentum) * self.train_option["eps"]
        for conn in self.edges:
            if conn not in self.ref_graph:
                self._g["dW"][conn] *= momentum
                self._g["dW"][conn] += \
                    eps * self._g["gW"][conn] / self.ref_count[conn]
                self.edges[conn].W += self._g["dW"][conn]
                if self.train_option["w_norm"] is not None:
                    w_norm_limit = self.train_option["w_norm"]
                    w_norm = (self.edges[conn].W ** 2).sum(0) + _EPS
                    w_norm = gnp.sqrt(w_norm_limit / w_norm)
                    w_norm = w_norm * (w_norm <= 1.) + (w_norm > 1.)
                    self.edges[conn].W *= w_norm

        for name in self.nodes:
            if name not in self.inputs:
                self._g["db"][name] *= momentum
                self._g["db"][name] += eps * self._g["gb"][name]
                self.nodes[name].b += self._g["db"][name]

    def feed_forward(self, data):
        y = {}
        N = data[data.keys()[0]].shape[0]
        dropout = self.train_option["dropout"] if self.is_training else 0
        for name in self.nodes:
            if name in self.inputs:
                y[name] = np.array(data[name])
            else:
                y[name] = np.zeros((N, self.nodes[name].size))
                if dropout > 0. and name not in self.outputs:
                    mask = np.random.rand(*y[name].shape) > dropout
                    y[name] *= mask

        for name in self.ff_order:
            if name not in self.inputs:
                y[name] = self.nodes[name].up(y[name])
            for name2 in self.ff_edges[name]:
                conn = (name, name2)
                y[name2] += self.edges[conn].up(y[name])
        return y

    def evaluate(self, data, measures):
        predicted = self.feed_forward(data)
        result = {}
        for name in measures:
            measure_func = measure_table[measures[name]]
            result[name] = measure_func(predicted[name], data[name])
        return result
