import os
import random
import time

import numpy as np
import cudamat.gnumpy as gnp
from easydict import EasyDict

import mang.visualization as vis
import mang.util as U
from mang.batch import Batch
import mang.node as mnode
from mang.feedforwardnetwork import FeedForwardNetwork as NN
from mang.autoencoder import AutoEncoder
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
        'noise': 'gaussian',
        'noise_level': 0.05,
        'eps': .1,
        'w_norm': 15.,
        'tied': False,
        }

class StackedAutoEncoder(object):
    def __init__(self, nodes, edges):
        self.K = [len(x) for x in nodes]
        self.pretrain_nodes = [[x for x in y] for y in nodes]
        self.pretrain_edges = edges
        n1 = sum([len(x) for x in nodes[:-1]])
        n2 = n1 + len(nodes[-1])
        self.top_nodes = range(n1, n2)
        nodes_1 = nodes
        nodes_2 = [[x for x in y] for y in nodes[:-1]]
        nodes_2.reverse()
        # I don't have any better idea
        for i in xrange(len(nodes_2[-1])):
            nodes_1[0][i] = mnode.InputNode(nodes_1[0][i].dim)
            nodes_2[-1][i].is_output = True
        self.nodes = [n for layer in (nodes_1 + nodes_2) for n in layer]
        edges_1 = edges
        edges_2 = [[(i2, i1) for (i1, i2) in e] for e in edges]
        edges_2.reverse()
        self.edges = []
        n_nodes = [len(x) for x in (nodes_1 + nodes_2)]
        edges_all = edges_1 + edges_2
        n1 = 0
        for (i, n) in enumerate(n_nodes[1:]):
            n2 = n1 + n_nodes[i]
            self.edges += [(i1 + n1, i2 + n2) \
                    for (i1, i2) in edges_all[i]]
            n1 = n2
        self.nn = NN(self.nodes, self.edges)

    def pretrain(self, inputs, **option):
        W1 = []
        W2 = []
        b1 = []
        b2 = []
        prefix = 'tmp_%d' % random.randint(0, 99999)
        try:
            print option
            data = [np.array(x) for x in inputs]
            for i in xrange(len(self.pretrain_nodes) - 1):
                if option['display']:
                    print 'Stacked AE: pre-training layer %d...' % (i + 1)
                U.save_pickle(option['pretrain_option'], 
                        '%s_option' % prefix)
                ae = AutoEncoder(self.pretrain_nodes[i + 1],
                        self.pretrain_nodes[i], self.pretrain_edges[i])
                U.save_pickle(ae, '%s_model' % prefix)
                U.save_pickle(data, '%s_data' % prefix)
                os.system('python pretrain_sae.py %s' % prefix)
                data = U.load_pickle('%s_output' % prefix)
                ae = U.load_pickle('%s_model' % prefix)
                half = len(ae.W)/2
                W1 += [np.array(x) for x in ae.W[:half]]
                W2 = [np.array(x) for x in ae.W[half:]] + W2
                i1 = ae.K[0]
                i2 = i1 + ae.K[1]
                b1 += [np.array(x) for x in ae.b[i1:i2]]
                b2 = [np.array(x) for x in ae.b[i2:]] + b2
                time.sleep(3)
            self.nn.W = W1 + W2
            self.nn.b = [np.zeros((1, 1))]*self.K[0] + b1 + b2
            if option['display']:
                print 'Stacked AE: pretraining complete!'
            os.system('rm %s*' % prefix)
        except AttributeError:
            os.system('rm %s*' % prefix)

    def _init_training(self, inputs, **option):
        option_nn = {}
        option_ = dict(_default_option)
        for key in option:
            option_[key] = option[key]
        option = dict(option_)
        option_nn['eps'] = option['eps']
        option_nn['noise'] = option['noise']
        option_nn['noise_param'] = option['noise_param']
        option_nn['momentum'] = option['momentum']
        option_nn['dropout'] = option['dropout']
        option_nn['n_epoch'] = option['n_epoch']
        option_nn['chunk_size'] = option['chunk_size']
        option_nn['batch_size'] = option['batch_size']
        option_nn['drop_view'] = option['drop_view']
        option_nn['w_norm'] = option['w_norm']
        option_nn['tied'] = []
        if option['tied']:
            option_nn['tied_t'] = []
        if option['pretrain']:
            self.pretrain(inputs, **option)
            option_nn['reset'] = False
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

    def fit(self, inputs, **option):
        for i_epoch in self.fit_iter(inputs, **option):
            pass

    def fit_iter(self, inputs, **option):
        option_nn = self._init_training(inputs, **option)
        outputs = [np.array(x) for x in inputs]
        for i_epoch in self.nn.fit_iter(inputs, outputs, **option_nn):
            yield i_epoch
        self._finish_training(**option)

    def fit_epoch(self, inputs, **option):
        outputs = [np.array(x) for x in inputs]
        self.nn.fit_epoch(inputs, outputs, **option)

    def transform(self, inputs):
        self.nn.feed_forward(inputs, output=self.top_nodes)

    def reconstruct(self, inputs):
        return self.nn.feed_forward(inputs)

    def evaluate(self, inputs, measures):
        predicted = self.reconstruct(inputs)
        K = len(inputs)
        return [measures[i](predicted[i], inputs[i]) for i in xrange(K)]
