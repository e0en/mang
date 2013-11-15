import time

import numpy as np
import mang.cudamat as cm

from mang.measure import MEASURE_TABLE
from mang.cost import D_COST_TABLE
from mang.noise import NOISE_TABLE
from mang import graph
from mang import node as mnode
from mang import edge as medge
from mang import util


_DEFAULT_OPTION = {
    'reset': True,
    'n_epoch': 300,
    'eps_decay': 0.998,
    'batch_size': 128,
    'callback': None,
    'edge_param': {},
    'node_param': {},
    'verbosity': 0,
    }

_DEFAULT_NODE_PARAM = {
    'eps': 1e-2,
    'init_b': 0,
    'momentum_i': 0.5,
    'momentum_f': 0.99,
    'dropout': 0.,
    'cost': None,
    'cost_param': None,
    'noise': None,
    'noise_param': None,
    'sparsity': 0.,
    'data_from': None,
    }

_DEFAULT_EDGE_PARAM = {
    'eps': 1e-2,
    'init_w': 1e-2,
    'w_norm': None,
    'momentum_i': 0.5,
    'momentum_f': 0.99,
    'decay': 0.,
    }

_EPS = 1e-6


class FeedForwardNetwork(object):
    """Feed-forward network trained using back-propagation."""

    def __init__(self, nodes, edges):
        self.nodes = {}
        for name in nodes:
            spec = nodes[name]
            node_type = mnode.NODE_TABLE[spec['type']]
            self.nodes[name] = node_type(spec['shape'], spec)

        self.edges = {}
        self.ref_count = {}
        for conn in edges:
            spec = edges[conn]
            if spec['type'] != 'ref':
                edge_type = medge.EDGE_TABLE[spec['type']]
                self.edges[conn] = edge_type(self.nodes, conn, spec)
                self.ref_count[conn] = 1.

        # edges that shares weights of other edges
        ref_edges = [x for x in edges if edges[x]['type'] == 'ref']
        self.ref_edges = {}
        for conn in ref_edges:
            spec = edges[conn]
            self.ref_edges[conn] = spec['original']
            self.ref_count[spec['original']] += 1
            edge_type = medge.EDGE_TABLE['ref']
            edge_original = self.edges[spec['original']]
            self.edges[conn] = edge_type(conn, edge_original, spec)

        self.real_edges = set(self.edges.keys()) - set(self.ref_edges.keys())

        (self.inputs, self.outputs) = graph.find_boundary(self.edges.keys())
        self.boundary = self.inputs | self.outputs

        # ff_edges: outgoing edges, bp_edges: incoming edges
        self.ff_edges = dict(
            [(name, [x[1] for x in self.edges if x[0] == name])
                for name in self.nodes])
        self.bp_edges = dict(
            [(name, [x[0] for x in self.edges if x[1] == name])
                for name in self.nodes])

        self.ff_order = graph.find_order(self.edges.keys())
        self.bp_order = [x for x in self.ff_order if x not in self.inputs]
        self.bp_order.reverse()

        self.is_training = False
        self.train_option = None
        self.edge_param = None
        self.node_param = None

    def _init_training(self, option):
        """Initialize training parameters."""

        # load default options
        _option = dict(_DEFAULT_OPTION)
        for key in option:
            _option[key] = option[key]
        option = _option

        used_gpu_memory = 0
        self.edge_param = dict(option['edge_param'])

        for conn in self.real_edges:
            param = dict(_DEFAULT_EDGE_PARAM)
            if conn in self.edge_param:
                for key in self.edge_param[conn]:
                    param[key] = self.edge_param[conn][key]
            self.edge_param[conn] = dict(param)
            if option['reset']:
                self.edges[conn].W = param['init_w'] * \
                    np.random.randn(*self.edges[conn].W.shape)
            self.edges[conn].init_training(option['batch_size'])
            used_gpu_memory += self.edges[conn].used_gpu_memory

        self.node_param = dict(option['node_param'])
        for name in self.nodes:
            param = dict(_DEFAULT_NODE_PARAM)
            if name in self.node_param:
                for key in self.node_param[name]:
                    param[key] = self.node_param[name][key]
            self.node_param[name] = dict(param)
            if self.nodes[name].use_bias and option['reset']:
                    self.nodes[name].b = \
                        param['init_b'] * np.ones(self.nodes[name].b.shape)
            self.nodes[name].init_training(option)
            used_gpu_memory += self.nodes[name].used_gpu_memory

        # temporary variables used for training
        # activations, and their gradients
        self._g = {'data': {}, 'db': {}, 'gb': {}, 'dW': {}, 'gW': {}, }

        # allocate spaces for data and biases
        cm.cublas_init()
        cm.CUDAMatrix.init_random()

        batch_size = option['batch_size']
        for name in self.boundary:
            data_from = self.node_param[name]['data_from']
            if data_from is None:
                self._g['data'][name] = \
                    cm.empty((batch_size, self.nodes[name].size))
                used_gpu_memory += 4 * batch_size * self.nodes[name].size

        for name in self.nodes:
            if self.nodes[name].use_bias:
                self._g['db'][name] = cm.empty(self.nodes[name].b.shape)
                self._g['gb'][name] = cm.empty(self.nodes[name].b.shape)
                self._g['db'][name].assign(0)
                self._g['gb'][name].assign(0)
                used_gpu_memory += 8 * self.nodes[name].b.shape[1]

        for conn in self.real_edges:
            self._g['dW'][conn] = cm.empty(self.edges[conn].W.shape)
            self._g['gW'][conn] = cm.empty(self.edges[conn].W.shape)
            self._g['dW'][conn].assign(0)
            self._g['gW'][conn].assign(0)
            used_gpu_memory += \
                8 * self.edges[conn].W.shape[0] * self.edges[conn].W.shape[1]

        # allocate spaces for dropout masks (still not sure if it's needed)
        self._g['mask'] = {}
        for name in self.nodes:
            if self.node_param[name]['dropout'] > 0.:
                self._g['mask'][name] = \
                    cm.empty((batch_size, self.nodes[name].size))
                used_gpu_memory += 4 * batch_size * self.nodes[name].size

        self.train_option = dict(option)
        self.is_training = True

        if self.train_option['verbosity'] > 0:
            print "Using %dMB of GPU memory" % (used_gpu_memory >> 20)

    def _finish_training(self):
        """store the trained parameters and remove GPU variables"""

        for conn in self.real_edges:
            self.edges[conn].finish_training()

        # un-link the tied edges.
        for conn in self.ref_edges:
            self.edges[conn] = self.edges[conn].materialize(self.nodes)
        self.ref_edges = {}
        self.real_edges = set(self.edges.keys())

        for name in self.nodes:
            self.nodes[name].finish_training()
            if self.node_param[name]['dropout'] > 0.:
                for name2 in self.ff_edges[name]:
                    conn = (name, name2)
                    self.edges[conn].W *= self.node_param[name]['dropout']

        del self._g, self.train_option, self.node_param, self.edge_param
        self.is_training = False

    def fit(self, data, **option):
        """Train network with given data and training options."""

        self._init_training(option)
        stat = []
        time_train = 0
        for i_epoch in xrange(self.train_option['n_epoch']):
            t_epoch_start = time.time()
            for conn in self.edge_param:
                self.edge_param[conn]['eps'] *= self.train_option['eps_decay']
            for name in self.node_param:
                self.node_param[name]['eps'] *= self.train_option['eps_decay']

            if i_epoch < 500:
                r_epoch = i_epoch / 500.
                for name in self.node_param:
                    momentum_f = self.node_param[name]['momentum_f']
                    momentum_i = self.node_param[name]['momentum_i']
                    self.node_param[name]['momentum'] = \
                        r_epoch * momentum_f + (1. - r_epoch) * momentum_i
                for conn in self.real_edges:
                    momentum_f = self.edge_param[conn]['momentum_f']
                    momentum_i = self.edge_param[conn]['momentum_i']
                    self.edge_param[conn]['momentum'] = \
                        r_epoch * momentum_f + (1. - r_epoch) * momentum_i
            else:
                for name in self.node_param:
                    self.node_param[name]['momentum'] = \
                        self.node_param[name]['momentum_f']
                for conn in self.edge_param:
                    self.edge_param[conn]['momentum'] = \
                        self.edge_param[conn]['momentum_f']

            n_iter = self.fit_epoch(data)
            time_train += time.time() - t_epoch_start
            stat += [{
                'epoch': i_epoch,
                'iter': n_iter,
                'time': time_train,
                'train_cost': None,
                'validation_cost': None,
                'train_measure': None,
                'validation_measure': None,
                'node_param': dict(self.node_param),
                'edge_param': dict(self.edge_param),
                }]

            if self.train_option['callback'] is not None:
                self.train_option['callback'](self, stat)

            # restore temporary GPU variables to prevent potential problems
            for name in self.nodes:
                self.nodes[name].to_gpu(self.train_option['batch_size'])
            for conn in self.real_edges:
                self.edges[conn].to_gpu(self.train_option['batch_size'])

        self._finish_training()

        return stat

    def fit_epoch(self, data):
        """Train network for one epoch (single sweep over training data)."""

        if isinstance(data, list):
            n_iter = 0
            for filename in data:
                data_item = util.load_file(filename)
                n_iter += self.fit_epoch(data_item)
            return n_iter

        n_sample = data[data.keys()[0]].shape[0]
        batch_size = self.train_option['batch_size']

        for name in data:
            data[name] = np.array(data[name], dtype=np.float32, order='F')

        # shuffle data
        rng_state = np.random.get_state()
        for name in data:
            np.random.set_state(rng_state)
            np.random.shuffle(data[name])

        n_batch = int(n_sample / batch_size)
        i_start = 0
        for _ in xrange(n_batch):
            i_end = i_start + batch_size
            for name in data:
                if name in self._g['data']:
                    self._g['data'][name].overwrite(data[name][i_start:i_end])
                if name in self.inputs:
                    self.nodes[name].y.overwrite(data[name][i_start:i_end])
            self.fit_step()  # mini-batch training
            i_start = i_end
        self._check_values()

        return n_batch

    def fit_step(self):
        """Perform one step of backpropagation learning."""

        # add noise to training data
        for name in self.inputs:
            if self.node_param[name]['noise'] is not None:
                noise_name = self.node_param[name]['noise']
                noise_param = self.node_param[name]['noise_param']
                noise_func = NOISE_TABLE[noise_name]
                noise_func(self.nodes[name].y, noise_param)

        self._feed_forward()
        self._back_propagate()
        self._update()

        '''
        # renormalize ReLU weights so that their maximum output is 1
        skip_nodes = set(self.boundary)
        for conn in self.ref_edges:
            skip_nodes |= set(conn)
            for conn2 in self.ref_edges:
                skip_nodes |= set(conn2)
        for name in self.bp_order:
            if not isinstance(self.nodes[name], mnode.ReLUNode):
                continue
            if name not in skip_nodes:
                shape_old = self.nodes[name].y.shape
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
        """
        Calculate activations of all nodes in network,
        using mini-batch stored in GPU as input.
        """

        for name in self.nodes:
            if name not in self.inputs:
                self.nodes[name].y.assign(0)
        for name in self.ff_order:
            if name not in self.inputs:
                self.nodes[name].up()
                # dropout: randomly drop hidden nodes using binary masks
                if self.is_training:
                    dropout = self.node_param[name]['dropout']
                    if dropout > 0.:
                        self._g['mask'][name].assign(1.)
                        self._g['mask'][name].dropout(dropout)
                        self.nodes[name].y.mult(self._g['mask'][name])
            for name2 in self.ff_edges[name]:
                conn = (name, name2)
                self.edges[conn].up(self.nodes)

        '''
        for name in self.nodes:
            data_from = self.node_param[name]['data_from']
            if data_from is not None:
                if data_from in self._g['data']:
                    self._g["data"][name].assign(self._g['data'][data_from])
                else:
                    self._g["data"][name].assign(self.nodes[data_from].y)
        '''

    def _back_propagate(self):
        """
        Back-propagate through network and calculate gradients of
        objective functions with respect to parameters.

        Todo: reduce the number of branches (how?).
        """

        for name in self.nodes:
            self.nodes[name].dy.assign(0)

        for name in self.outputs:
            # assign ground-truth values
            data_from = self.node_param[name]['data_from']
            if data_from is not None:
                if data_from in self._g['data']:
                    truth = self._g['data'][data_from]
                else:
                    truth = self.nodes[data_from].y
            else:
                truth = self._g['data'][name]

            # apply cost functions to output nodes
            cost_name = self.node_param[name]['cost']
            if cost_name is None:
                # zero error for non-data output nodes.
                if self.node_param[name]['sparsity'] > 0:
                    lamb = -self.node_param[name]['sparsity']
                    self.nodes[name].dy.add_mult_sign(self.nodes[name].y, lamb)
                self.nodes[name].down()
            # handle special cases for faster calculation
            elif isinstance(self.nodes[name], mnode.SoftmaxNode) or \
                    (isinstance(self.nodes[name], mnode.LogisticNode) and
                        cost_name == 'cross_entropy'):
                truth.subtract(self.nodes[name].y, self.nodes[name].dy)
            else:
                cost_func = D_COST_TABLE[cost_name]
                cost_func(self.nodes[name].y, truth, self.nodes[name].dy)
                self.nodes[name].down()

        for conn in self.real_edges:
            self._g['gW'][conn].assign(0)
        for name in self.bp_order:
            if name not in self.outputs:
                if self.node_param[name]['sparsity'] > 0:
                    lamb = -self.node_param[name]['sparsity']
                    self.nodes[name].dy.add_mult_sign(self.nodes[name].y, lamb)
                self.nodes[name].down()
            if self.nodes[name].use_bias:
                self.nodes[name].gradient(self._g['gb'][name])
            # dropout: randomly drop hidden nodes using binary masks
            dropout = self.node_param[name]['dropout']
            if dropout > 0. and name not in self.outputs:
                self.nodes[name].dy.mult(self._g['mask'][name])
            for name2 in self.bp_edges[name]:
                conn = (name2, name)
                if conn in self.ref_edges:
                    conn2 = self.ref_edges[conn]
                else:
                    conn2 = conn
                self.edges[conn].gradient(self.nodes, self._g['gW'][conn2])
                self.edges[conn].down(self.nodes)

    def _update(self):
        """
        Update parameters using the gradients calculated in _back_propagate.
        """

        for conn in self.real_edges:
            momentum = self.edge_param[conn]['momentum']
            eps = (1. - momentum) * self.edge_param[conn]['eps']
            decay = self.edge_param[conn]['decay']
            self._g['dW'][conn].mult(momentum)
            self._g['dW'][conn].add_mult(
                self._g['gW'][conn], eps / self.ref_count[conn])
            if decay > 0:
                self._g['dW'][conn].add_mult(self.edges[conn].W, - eps * decay)
            self.edges[conn].W.add(self._g['dW'][conn])

            # limit the squared norm of features
            if self.edge_param[conn]['w_norm'] is not None:
                w_norm_limit = self.edge_param[conn]['w_norm']
                if isinstance(self.edges[conn], medge.ConvolutionalEdge):
                    self.edges[conn].W.norm_limit(w_norm_limit, 1)
                elif isinstance(self.edges[conn], medge.DeconvolutionalEdge):
                    self.edges[conn].W.norm_limit(w_norm_limit, 1)
                else:
                    self.edges[conn].W.norm_limit(w_norm_limit, 0)

        for name in self.nodes:
            if name not in self.inputs and self.nodes[name].use_bias:
                momentum = self.node_param[name]['momentum']
                eps = (1. - momentum) * self.node_param[name]['eps']
                self._g['db'][name].mult(momentum)
                self._g['db'][name].add_mult(self._g['gb'][name], eps)
                self.nodes[name].b.add(self._g['db'][name])

    def _check_values(self):
        """Check if there is any NaN/inf in model parameters."""

        for conn in self.real_edges:
            assert np.isinf(self.edges[conn].W.asarray()).sum() == 0
            assert np.isnan(self.edges[conn].W.asarray()).sum() == 0
        for name in self.nodes:
            if name not in self.inputs and self.nodes[name].use_bias:
                assert np.isinf(self.nodes[name].b.asarray()).sum() == 0
                assert np.isnan(self.nodes[name].b.asarray()).sum() == 0

    def feed_forward(self, data, batch_size=128, nodes=None):
        """Calculate output node activations from input data."""

        for name in data:
            data[name] = np.array(data[name], dtype=np.float32, order='F')
        n_sample = data[data.keys()[0]].shape[0]
        result = {}
        if nodes is None:
            nodes = self.outputs
        for name in nodes:
            result[name] = np.zeros((n_sample, self.nodes[name].size))

        # initialize GPU variables of nodes and edges
        if self.is_training:
            batch_size = self.train_option['batch_size']
        for name in self.nodes:
            self.nodes[name].to_gpu(batch_size)
        for conn in self.real_edges:
            self.edges[conn].to_gpu(batch_size)

        n_batch = int(np.ceil(1. * n_sample / batch_size))
        for i_batch in xrange(n_batch):
            i_start = i_batch * batch_size
            i_end = min(n_sample, i_start + batch_size)
            real_batch_size = i_end - i_start
            for name in self.inputs:
                tmp = np.zeros((batch_size, self.nodes[name].size),
                               dtype=np.float32, order='F')
                tmp[:real_batch_size] = data[name][i_start:i_end]
                self.nodes[name].y.overwrite(tmp)
            self._feed_forward()
            for name in result:
                result[name][i_start:i_end] = \
                        self.nodes[name].y.asarray()[:real_batch_size]

        for name in self.nodes:
            self.nodes[name].from_gpu()
        for conn in self.real_edges:
            self.edges[conn].from_gpu()

        return result

    def evaluate(self, data, measures, option={}, n_max=None):
        """Evaluate performance of network using data and measure functions."""

        for key in data:
            data[key] = np.array(data[key], dtype=np.float32, order='F')
            if n_max is not None:
                data[key] = data[key][:n_max]
        predicted = self.feed_forward(data)
        result = {}
        for name in measures:
            measure_func = MEASURE_TABLE[measures[name]]
            result[name] = measure_func(predicted[name], data[name], **option)
        return result

    def save(self, filename):
        """Save current model as a file."""

        result = {
            # save node/edge parameters
            'nodes': dict([(x, self.nodes[x].to_dict()) for x in self.nodes]),
            'edges': dict([(x, self.edges[x].to_dict()) for x in self.edges]),

            # save topology
            'inputs': list(self.inputs),
            'outputs': list(self.outputs),
            'ref_edges': self.ref_edges,
            'ff_edges': self.ff_edges,
            'bp_edges': self.bp_edges,
            'ff_order': self.ff_order,
            'bp_order': self.bp_order,

            # additional information
            'ref_count': self.ref_count,
            }
        util.save_msgpack(filename, result)

    @staticmethod
    def load(filename):
        """Load a network from file and return the network object."""

        net = FeedForwardNetwork({}, {})
        result = util.load_msgpack(filename)

        net.inputs = set(result['inputs'])
        net.outputs = set(result['outputs'])
        net.ref_edges = result['ref_edges']

        net.ff_edges = result['ff_edges']
        net.bp_edges = result['bp_edges']

        net.ff_order = result['ff_order']
        net.bp_order = result['bp_order']

        net.ref_count = result['ref_count']
        net.is_training = False

        net.nodes = {}
        for name in result['nodes']:
            info = result['nodes'][name]
            net.nodes[name] = mnode.NODE_TABLE[info['type']].from_dict(info)

        net.real_edges = \
            set(result['edges'].keys()) - set(net.ref_edges.keys())

        for conn in net.real_edges:
            info = result['edges'][conn]
            net.edges[conn] = \
                medge.EDGE_TABLE[info['type']].from_dict(net.nodes, info)

        for conn in net.ref_edges:
            info = result['edges'][conn]
            original = net.edges[net.ref_edges[conn]]
            net.edges[conn] = \
                medge.EDGE_TABLE['ref'].from_dict(original, info)

        return net
