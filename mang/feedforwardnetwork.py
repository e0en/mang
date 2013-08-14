import numpy as np
import cudamat.gnumpy as gnp
from easydict import EasyDict

from mang.measure import measure_table
from mang.cost import cost_table
from mang import graph
from mang import util as U
from mang import node as mnode
from mang import edge as medge

_default_option = {
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
        "shared": {}, # shared weight pairs
        "shared_t": {}, # shared weight pairs with transpose
        "noise": None,
        "noise_param": None,
        "measure": None,
        "cost": None,
        "cost_scale": None,
        "callback": None,
        }

class FeedForwardNetwork(object):
    def __init__(self, nodes, edges):
        self.nodes = {}
        for name in nodes:
            spec = nodes[name]
            node_type = mnode.node_table[spec["type"]]
            if "option" not in spec:
                spec["option"] = {}
            self.nodes[name] = node_type(spec["shape"], spec["option"])
        self.n_node = len(self.nodes.keys())

        self.edges = {}
        for conn in edges:
            spec = edges[conn]
            edge_type = medge.edge_table[spec["type"]]
            if "option" not in spec:
                spec["option"] = {}

            node_in = self.nodes[conn[0]]
            node_out = self.nodes[conn[1]]
            self.edges[conn] = edge_type(node_in, node_out, spec["option"])
        self.n_edge = len(self.edges.keys())

        (self.inputs, self.outputs) = graph.find_boundary(self.edges.keys())
        self.boundary = self.inputs | self.outputs
        self.n_input = len(self.inputs)
        self.n_output = len(self.outputs)

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

    def _init_training(self, option):
        # load default options
        _option = dict(_default_option)
        for key in option:
            _option[key] = option[key]
        option = _option

        option["skip_list"] = option["shared"].keys() +\
                option["shared_t"].keys()

        if len(self.inputs) == 1:
            option["drop_view"] = False

        for conn in self.edges:
            if conn not in option["skip_list"]:
                self.edges[conn].init_training(option)
        for name in self.nodes:
            self.nodes[name].init_training(option)

        if option["cost"] is None:
            # squared error is the default cost function
            option["cost"] = {}
            for name in self.outputs:
                option["cost"][name] = cost_table["squared_error"]

        if option["measure"] is None:
            # RMSE is the default performance measure
            option["measure"] = {}
            for name in self.outputs:
                option["measure"][name] = measure_table["rmse"]

        # temporary variables used for training
        # activations, and their gradients
        self._g = {"data": {}, "y": {}, "do": {}, }

        # allocate spaces for data and biases
        batch_size = option["batch_size"]
        for name in self.boundary:
            self._g["data"][name] =\
                    gnp.zeros((batch_size, self.nodes[name].size))
        for name in self.nodes:
            self._g["y"][name] =\
                    gnp.zeros((batch_size, self.nodes[name].size))
            self._g["do"][name] =\
                    gnp.zeros((batch_size, self.nodes[name].size))

        # allocate spaces for dropout masks (still not sure if it's needed)
        if option["dropout"] > 0.:
            self._g["mask"] = {}
            for name in self.nodes:
                self._g["mask"][name] =\
                        gnp.zeros((batch_size, self.nodes[name].size))

        if option["drop_view"]:
            self._g["drop_mask"] = gnp.zeros((self.n_input + 1, batch_size))

        self._g = EasyDict(self._g)
        return option

    def _copy_params(self, option):
        for conn in self.edges:
            if conn not in option["skip_list"]:
                self.edges[conn].copy_params()
        for conn in self.edges:
            if conn in option["shared"]:
                edge2 = self.edges[option["shared"][conn]]
                self.edges[conn].W = np.array(edge2.W)
            elif conn in option["shared_t"]:
                edge2 = self.edges[option["shared_t"][conn]]
                self.edges[conn].W = np.array(edge2.W.T)
        for name in self.nodes:
            self.nodes[name].copy_params()
            if option["dropout"] > 0.:
                if name not in self.inputs:
                    for name2 in self.ff_edges[name]:
                        conn = (name, name2)
                        self.edges[conn].W *= option["dropout"]

    # store the trained parameters and remove GPU variables
    def _finish_training(self, option):
        self._copy_params(option)
        for conn in self.edges:
            if conn not in option["skip_list"]:
                self.edges[conn].finish_training()
        for name in self.nodes:
            self.nodes[name].finish_training()
        del self._g

        gnp.free_reuse_cache()
        reload(gnp)

    def fit(self, data, **option):
        option = dict(self._init_training(option))
        momentum_f = option["momentum"]
        measures = option["measure"]
        for i_epoch in xrange(option["n_epoch"]):
            option["eps"] *= 0.998
            if i_epoch < 500:
                r_epoch = i_epoch / 500.
                option["momentum"] = r_epoch * momentum_f +\
                                    0.5 * (1. - r_epoch)
            else:
                momentum = momentum_f
            self.fit_epoch(data, option)
            if option["callback"] != None:
                self._copy_params(option)
                option["callback"](self, i_epoch)
        self._finish_training(option)

    def fit_epoch(self, data, option):
        N = data[data.keys()[0]].shape[0]
        assert set(self.boundary) == set(data.keys())

        # shuffle data
        rng_state = np.random.get_state()
        for name in data:
            np.random.set_state(rng_state)
            np.random.shuffle(data[name])

        n_batch = int(N / option["batch_size"])
        i1 = 0
        for i_batch in xrange(n_batch):
            i2 = i1 + option["batch_size"]
            for name in data:
                self._g.data[name] = gnp.garray(data[name][i1:i2])

            if option["noise"]:
                # add pre-specified noise pattern to input samples
                for name in self.inputs:
                    noise_func = noise_table[option["noise"][name]]
                    param = option["noise_param"][name]
                    self._g.data[name] =\
                            noise_func(self._g.data[name], param)
            if option["drop_view"]:
                # randomly drop views if user chose to do so
                self._g.drop_mask = gnp.rand(self.n_input + 1,
                        option["batch_size"])
                prob = 1. / (self.n_input + 1)
                for (i, name) in enumerate(self.inputs):
                    tmp = self._g.drop_mask[self.n_input] <= prob
                    tmp += self._g.drop_mask[i] <= prob
                    tmp >= 0
                    self._g.data[name] = (self._g.data[name].T * tmp).T

            # mini-batch training
            self.fit_step(option)

            i1 = i2

    # calculate the gradient using BP and update parameters
    def fit_step(self, option):
        self._feed_forward(option)
        self._back_propagate(option)
        self._update(option)

    def _feed_forward(self, option):
        for name in self.nodes:
            if name in self.inputs:
                self._g.y[name] = self._g.data[name]
            else:
                self._g.y[name] *= 0
        for name in self.ff_order:
            if name not in self.inputs:
                self._g.y[name] = self.nodes[name].up(self._g.y[name])
                # dropout: randomly drop hidden nodes using binary masks
                if option["dropout"] > 0. and name not in self.outputs:
                    self._g.mask[name] = gnp.rand(*self._g.y[name].shape) >\
                            option["dropout"]
                    self._g.y[name] *= self._g.mask[name]

            for name2 in self.ff_edges[name]:
                conn = (name, name2)
                if conn in option["shared"]:
                    conn2 = option["shared"][conn]
                    self._g.y[name2] +=\
                            self.edges[conn2].up(self._g.y[name])
                elif conn in option["shared_t"]:
                    conn2 = option["shared_t"][conn]
                    self._g.y[name2] +=\
                            self.edges[conn2].down(self._g.y[name])
                else:
                    self._g.y[name2] += self.edges[conn].up(self._g.y[name])

    def _back_propagate(self, option):
        for name in self.nodes:
            if name not in self.boundary:
                self._g.do[name] *= 0
            elif name in self.outputs:
                cost_func = cost_table[option["cost"][name]]
                self._g.do[name] = cost_func(self._g.y[name],
                                            self._g.data[name], is_d=True)
        for conn in self.edges:
            if conn not in option["skip_list"]:
                self.edges[conn].gW *= 0

        for name in self.bp_order:
            if name not in self.inputs:
                self.nodes[name].down(self._g.y[name], self._g.do[name])
                self.nodes[name].gradient(self._g.do[name])
                # dropout: randomly drop hidden nodes using binary masks
                if option["dropout"] > 0. and name not in self.outputs:
                    self._g.do[name] *= self._g.mask[name]
            for name2 in self.bp_edges[name]:
                if name2 in self.inputs:
                    continue

                conn = (name2, name)
                if conn in option["shared_t"]:
                    conn2 = option["shared_t"][conn]
                    self.edges[conn2].gradient(self._g.do[name],
                            self._g.y[name2])
                    self._g.do[name2] +=\
                            self.edges[conn2].up(self._g.do[name])
                else:
                    if conn in option["shared"]:
                        conn2 = option["shared"][conn]
                    else:
                        conn2 = conn
                    self.edges[conn2].gradient(self._g.y[name2],
                            self._g.do[name])
                    self._g.do[name2] +=\
                            self.edges[conn2].down(self._g.do[name])

    def _update(self, option):
        for conn in self.edges:
            if conn not in option["skip_list"]:
                self.edges[conn].update(option)
        for name in self.nodes:
            if name not in self.inputs:
                self.nodes[name].update(option)

    def feed_forward(self, data, **option):
        y = {}
        N = data[data.keys()[0]].shape[0]
        for name in self.nodes:
            if name in self.inputs:
                y[name] = np.array(data[name])
            else:
                y[name] = np.zeros((N, self.nodes[name].size))
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
