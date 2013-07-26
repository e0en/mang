import numpy as np
import cudamat.gnumpy as gnp
from easydict import EasyDict

import mang.node as mnode
from mang.batch import Batch

from mang.feedforwardnetwork import FeedForwardNetwork as NN
from mang.data import Data
from mang import measure

_default_option = {
        'reset': True,
        'initW': 0.01,
        'n_epoch': 1000,
        'batch_size': 100,
        'chunk_size': 10000,
        'dropout': 0.5,
        'drop_view': True,
        'display': True,
        'momentum': 0.99,
        'noise': None,
        'noise_param': None,
        'eps': .1,
        'w_norm': 15.,
        'tied': False,
        'cost': None,
        'cost_scale': None,
        }

class AutoEncoder(object):
    def __init__(self, hidden_nodes, output_nodes, edges):
        input_nodes = [mnode.InputNode(x.dim) for x in output_nodes]
        for i in xrange(len(output_nodes)):
            output_nodes[i].is_output = True
        self.K = [len(output_nodes), len(hidden_nodes), len(output_nodes)]
        self.nodes = input_nodes + hidden_nodes + output_nodes
        n1 = len(output_nodes)
        n2 = n1 + len(hidden_nodes)
        self.top_nodes = range(n1, n2)
        edges_1 = []
        edges_2 = []
        for (i1, i2) in edges:
            edges_1 += [(i1, n1 + i2)]
            edges_2 += [(n1 + i2, n2 + i1)]
        self.edges = edges_1 + edges_2
        self.nn = NN(self.nodes, self.edges)

    def _init_training(self, **option):
        option_nn = {}
        option_ = dict(_default_option)
        for key in option:
            option_[key] = option[key]
        option_nn['eps'] = option_['eps']
        option_nn['noise'] = option_['noise']
        option_nn['noise_param'] = option_['noise_param']
        option_nn['momentum'] = option_['momentum']
        option_nn['dropout'] = option_['dropout']
        option_nn['n_epoch'] = option_['n_epoch']
        option_nn['chunk_size'] = option_['chunk_size']
        option_nn['batch_size'] = option_['batch_size']
        option_nn['drop_view'] = option_['drop_view']
        option_nn['w_norm'] = option_['w_norm']
        option_nn['cost'] = option_['cost']
        option_nn['cost_scale'] = option_['cost_scale']
        if option_['tied']:
            half = len(self.edges)/2
            option_nn['tied_t'] = [(i, i + half) for i in xrange(half)]
        option_nn = self.nn._init_training(**option_nn)
        return option_nn

    def _finish_training(self, **option):
        pass

    @property
    def W(self):
        return self.nn.W

    @property
    def b(self):
        return self.nn.b

    def fit(self, input_data, **option):
        for i_epoch in self.fit_iter(input_data, **option):
            pass

    def fit_iter(self, input_data, **option):
        option_nn = self._init_training(**option)
        output_data = [Data(x) for x in input_data]
        for i_epoch in self.nn.fit_iter(input_data, output_data, \
                **option_nn):
            yield i_epoch
        self._finish_training(**option)

    def transform(self, inputs):
        return self.nn.feed_forward(inputs, output=self.top_nodes)

    def reconstruct(self, inputs):
        return self.nn.feed_forward(inputs)

    def evaluate(self, inputs, measures):
        predicted = self.reconstruct(inputs)
        K = len(inputs)
        return [measures[i](predicted[i], inputs[i]) for i in xrange(K)]
