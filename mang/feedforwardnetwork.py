import time

import numpy as np
import mang.cudamat as cm

from mang.measure import MEASURE_TABLE
from mang.cost import D_COST_TABLE
from mang import graph
from mang import node as mnode
from mang import edge as medge


_DEFAULT_OPTION = {
    "reset": True,
    "n_epoch": 300,
    "eps_decay": 0.998,
    "batch_size": 128,
    "callback": None,
    "edge_param": {},
    "node_param": {},
    }

_DEFAULT_NODE_PARAM = {
    "eps": 1e-2,
    "init_b": 0,
    "momentum_i": 0.5,
    "momentum_f": 0.99,
    "dropout": 0.,
    "cost": "squared_error",
    "noise": None,
    "noise_param": None,
    }

_DEFAULT_EDGE_PARAM = {
    "eps": 1e-2,
    "init_w": 1e-2,
    "w_norm": 15.,
    "momentum_i": 0.5,
    "momentum_f": 0.99,
    "decay": 0.,
    }

_EPS = 1e-6


class FeedForwardNetwork(object):
    """Feed-forward network trained using back-propagation."""

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
                self.edges[conn] = \
                    edge_type(node_in.shape, node_out.shape, spec)
                self.ref_count[conn] = 1.

        # edges that shares weights of other edges
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
        """Initialize training parameters."""

        # load default options
        _option = dict(_DEFAULT_OPTION)
        for key in option:
            _option[key] = option[key]
        option = _option

        self.edge_param = dict(option["edge_param"])
        for conn in self.edges:
            if conn not in self.ref_graph:
                param = dict(_DEFAULT_EDGE_PARAM)
                if conn in self.edge_param:
                    for key in self.edge_param[conn]:
                        param[key] = self.edge_param[conn][key]
                self.edge_param[conn] = dict(param)
                self.edges[conn].W = param["init_w"] * \
                    np.random.randn(*self.edges[conn].W.shape)
                self.edges[conn].init_training(option)

        self.node_param = dict(option["node_param"])
        for name in self.nodes:
            param = dict(_DEFAULT_NODE_PARAM)
            if name in self.node_param:
                for key in self.node_param[name]:
                    param[key] = self.node_param[name][key]
            self.node_param[name] = dict(param)
            self.nodes[name].b = \
                param["init_b"] * np.ones(self.nodes[name].b.shape)
            self.nodes[name].init_training(option)

        # temporary variables used for training
        # activations, and their gradients
        self._g = {"data": {}, "db": {}, "gb": {}, "dW": {}, "gW": {}, }

        # allocate spaces for data and biases
        cm.cublas_init()
        cm.CUDAMatrix.init_random()

        batch_size = option["batch_size"]
        for name in self.boundary:
            self._g["data"][name] = \
                cm.empty((batch_size, self.nodes[name].size))

        for name in self.nodes:
            self._g["db"][name] = cm.empty(self.nodes[name].b.shape)
            self._g["gb"][name] = cm.empty(self.nodes[name].b.shape)
            self._g["db"][name].assign(0)
            self._g["gb"][name].assign(0)

        for conn in self.edges:
            if conn not in self.ref_graph:
                self._g["dW"][conn] = cm.empty(self.edges[conn].W.shape)
                self._g["gW"][conn] = cm.empty(self.edges[conn].W.shape)
                self._g["dW"][conn].assign(0)
                self._g["gW"][conn].assign(0)

        # allocate spaces for dropout masks (still not sure if it's needed)
        self._g["mask"] = {}
        for name in self.nodes:
            if self.node_param[name]["dropout"] > 0.:
                self._g["mask"][name] = \
                    cm.empty((batch_size, self.nodes[name].size))

        self.train_option = dict(option)
        self.is_training = True

    def _finish_training(self):
        """store the trained parameters and remove GPU variables"""
        for conn in self.edges:
            if conn not in self.ref_graph:
                self.edges[conn].finish_training()
        for conn in self.ref_graph:
            self.edges[conn] = self.edges[conn].materialize()

        for name in self.nodes:
            self.nodes[name].finish_training()
            if self.node_param[name]["dropout"] > 0.:
                for name2 in self.ff_edges[name]:
                    conn = (name, name2)
                    self.edges[conn].W *= self.node_param[name]["dropout"]

        del self._g, self.train_option, self.node_param, self.edge_param
        self.is_training = False

    def fit(self, data, **option):
        """Train network with given data and training options."""

        self._init_training(option)
        stat = []
        time_train = 0
        for i_epoch in xrange(self.train_option["n_epoch"]):
            t_epoch_start = time.time()
            for conn in self.edge_param:
                self.edge_param[conn]["eps"] *= self.train_option["eps_decay"]
            for name in self.node_param:
                self.node_param[name]["eps"] *= self.train_option["eps_decay"]

            if i_epoch < 500:
                r_epoch = i_epoch / 500.
                for name in self.node_param:
                    momentum_f = self.node_param[name]["momentum_f"]
                    momentum_i = self.node_param[name]["momentum_i"]
                    self.node_param[name]["momentum"] = \
                        r_epoch * momentum_f + (1. - r_epoch) * momentum_i
                for conn in self.edge_param:
                    momentum_f = self.edge_param[conn]["momentum_f"]
                    momentum_i = self.edge_param[conn]["momentum_i"]
                    self.edge_param[conn]["momentum"] = \
                        r_epoch * momentum_f + (1. - r_epoch) * momentum_i
            else:
                for name in self.node_param:
                    self.node_param[name]["momentum"] = \
                        self.node_param[name]["momentum_f"]
                for conn in self.edge_param:
                    self.edge_param[conn]["momentum"] = \
                        self.edge_param[conn]["momentum_f"]

            self.fit_epoch(data)
            time_train += time.time() - t_epoch_start
            n = data[data.keys()[0]].shape[0]
            n_batch = int(n / self.train_option["batch_size"])
            stat += [{
                "epoch": i_epoch,
                "iter": i_epoch * n_batch,
                "time": time_train,
                "train_cost": None,
                "validation_cost": None,
                "train_measure": None,
                "validation_measure": None,
                "node_param": dict(self.node_param),
                "edge_param": dict(self.edge_param),
                }]

            if self.train_option["callback"] is not None:
                self.train_option["callback"](self, stat)

            for name in self.nodes:
                self.nodes[name].to_gpu(self.train_option["batch_size"])
        self._finish_training()

    def fit_epoch(self, data):
        """Train network for one epoch (single sweep over training data)."""

        N = data[data.keys()[0]].shape[0]
        batch_size = self.train_option["batch_size"]
        assert set(self.boundary) == set(data.keys())

        for name in data:
            data[name] = np.array(data[name], dtype=np.float32, order="F")

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
                if name in self.inputs:
                    self.nodes[name].y.overwrite(data[name][i1:i2])
                elif name in self.outputs:
                    self._g["data"][name].overwrite(data[name][i1:i2])
            self.fit_step()  # mini-batch training
            i1 = i2
        self._check_values()

    def fit_step(self):
        """Perform one step of backpropagation learning."""

        self._feed_forward()
        self._back_propagate()
        self._update()

        '''
        # renormalize ReLU weights so that their maximum output is 1
        skip_nodes = set(self.boundary)
        for conn in self.ref_graph:
            skip_nodes |= set(conn)
            for conn2 in self.ref_graph:
                skip_nodes |= set(conn2)
        for name in self.bp_order:
            if not isinstance(self.nodes[name], mnode.ReLUNode):
                continue
            if name not in skip_nodes:
                shape_old = self._g["y"][name].shape
                self.nodes[name].y.reshape((shape_old[0] * shape_old[1], 1))
                y_max = float(self.nodes[name].y.max(0).asarray())
                self.nodes[name].y.reshape(shape_old)
                if y_max <= 1.:
                    continue
                self.nodes[name].b.divide(y_max)
                for name2 in self.bp_edges[name]:
                    self.edges[name2, name].W.divide(y_max)
                for name2 in self.ff_edges[name]:
                    self.edges[name, name2].W.mult(y_max)
        '''

    def _feed_forward(self):
        for name in self.nodes:
            if name not in self.inputs:
                self.nodes[name].y.assign(0)
        for name in self.ff_order:
            if name not in self.inputs:
                self.nodes[name].up()
                # dropout: randomly drop hidden nodes using binary masks
                dropout = self.node_param[name]["dropout"]
                if self.is_training and dropout > 0.:
                    self._g["mask"][name].assign(1.)
                    self._g["mask"][name].dropout(dropout)
                    self.nodes[name].y.mult(self._g["mask"][name])
            for name2 in self.ff_edges[name]:
                conn = (name, name2)
                self.edges[conn].up(self.nodes[name].y, self.nodes[name2].y)

    def _back_propagate(self):
        """
        Back-propagate through network and calculate gradients of
        objective functions with respect to parameters.

        Todo: reduce the number of branches (how?).
        """

        for name in self.nodes:
            if name not in self.boundary:
                self.nodes[name].dy.assign(0)
            elif name in self.outputs:
                cost_name = self.node_param[name]["cost"]

                # handle special cases for faster calculation
                if isinstance(self.nodes[name], mnode.SoftmaxNode) or \
                        (isinstance(self.nodes[name], mnode.LogisticNode) and
                            cost_name == "cross_entropy"):
                    self._g["data"][name].subtract(
                        self.nodes[name].y, self.nodes[name].dy)
                else:
                    cost_func = D_COST_TABLE[cost_name]
                    cost_func(self.nodes[name].y, self._g["data"][name],
                              self.nodes[name].dy)
                    self.nodes[name].down()

        for conn in self.edges:
            if conn not in self.ref_graph:
                self._g["gW"][conn].assign(0)
        for name in self.bp_order:
            if name not in self.outputs:
                self.nodes[name].down()
            if name not in self.inputs:
                self.nodes[name].gradient(self._g["gb"][name])
                # dropout: randomly drop hidden nodes using binary masks
                dropout = self.node_param[name]["dropout"]
                if dropout > 0. and name not in self.outputs:
                    self.nodes[name].dy.mult(self._g["mask"][name])
            for name2 in self.bp_edges[name]:
                conn = (name2, name)
                if conn in self.ref_graph:
                    conn2 = self.ref_graph[conn]
                else:
                    conn2 = conn
                self.edges[conn].gradient(
                    self.nodes[name2].y, self.nodes[name].dy,
                    self._g["gW"][conn2])
                if name2 not in self.inputs:
                    self.edges[conn].down(
                        self.nodes[name].dy, self.nodes[name2].dy,
                        self.nodes[name].y, self.nodes[name2].y)

    def _update(self):
        """
        Update parameters using the gradients calculated in _back_propagate.
        """

        for conn in self.edges:
            if conn not in self.ref_graph:
                momentum = self.edge_param[conn]["momentum"]
                eps = (1. - momentum) * self.edge_param[conn]["eps"]
                decay = self.edge_param[conn]["decay"]
                self._g["dW"][conn].mult(momentum)
                self._g["dW"][conn].add_mult(self._g["gW"][conn],
                                             eps / self.ref_count[conn])
                if decay > 0:
                    self._g["dW"][conn].add_mult(self.edges[conn].W,
                                                 - eps * decay)
                self.edges[conn].W.add(self._g["dW"][conn])
                if self.edge_param[conn]["w_norm"] is not None:
                    w_norm_limit = self.edge_param[conn]["w_norm"]
                    if isinstance(self.edges[conn], medge.ConvolutionalEdge):
                        axis = 1
                    else:
                        axis = 0
                    self.edges[conn].W.norm_limit(w_norm_limit, axis)

        for name in self.nodes:
            if name not in self.inputs:
                momentum = self.node_param[name]["momentum"]
                eps = (1. - momentum) * self.node_param[name]["eps"]
                self._g["db"][name].mult(momentum)
                self._g["db"][name].add_mult(self._g["gb"][name], eps)
                self.nodes[name].b.add(self._g["db"][name])

    def _check_values(self):
        for conn in self.edges:
            if conn not in self.ref_graph:
                assert np.isinf(self.edges[conn].W.asarray()).sum() == 0
                assert np.isnan(self.edges[conn].W.asarray()).sum() == 0
        for name in self.nodes:
            if name not in self.inputs and self.nodes[name].use_bias:
                assert np.isinf(self.nodes[name].b.asarray()).sum() == 0
                assert np.isnan(self.nodes[name].b.asarray()).sum() == 0

    def feed_forward(self, data, batch_size=128):
        for name in data:
            data[name] = np.array(data[name], dtype=np.float32, order="F")
        N = data[data.keys()[0]].shape[0]
        result = {}
        for name in self.outputs:
            result[name] = np.zeros((N, self.nodes[name].size))
        for name in self.nodes:
            self.nodes[name].to_gpu(batch_size)
        n_batch = int(np.ceil(N / batch_size))
        for i_batch in xrange(n_batch):
            i1 = i_batch * batch_size
            i2 = min(N, i1 + batch_size)
            n_sample = i2 - i1
            for name in self.inputs:
                tmp = np.zeros((batch_size, self.nodes[name].size),
                               dtype=np.float32, order="F")
                tmp[:n_sample] = data[name][i1:i2]
                self.nodes[name].y.overwrite(tmp)
            self._feed_forward()
            for name in self.outputs:
                result[name][i1:i2] = self.nodes[name].y.asarray()
        for name in self.nodes:
            self.nodes[name].from_gpu()
        return result

    def evaluate(self, data, measures, n_max=None):
        for key in data:
            data[key] = np.array(data[key], dtype=np.float32, order="F")
            if n_max is not None:
                data[key] = data[key][:n_max]
        predicted = self.feed_forward(data)
        result = {}
        for name in measures:
            measure_func = MEASURE_TABLE[measures[name]]
            result[name] = measure_func(predicted[name], data[name])
        return result
