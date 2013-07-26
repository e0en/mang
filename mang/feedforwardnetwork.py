import numpy as np
import cudamat.gnumpy as gnp
from easydict import EasyDict

import mang.visualization as vis
import mang.util as U
import mang.node as mnode
from mang.batch import Batch
from mang import measure
from mang import cost
from mang.data import Data

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
        'eps': .1,
        'w_norm': 15.,
        'tied': [], # tied weight pairs
        'tied_t': [], # tied weight pairs with transpose
        'noise': None,
        'noise_param': None,
        'measure': None,
        'cost': None,
        'cost_scale': None,
        }

class FeedForwardNetwork(object):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.n_node = len(self.nodes)
        self.inputs = [i for i in range(self.n_node) \
                if isinstance(self.nodes[i], mnode.InputNode)]
        self.n_input = len(self.inputs)
        self.outputs = [i for i in range(self.n_node) \
                if self.nodes[i].is_output]
        self.n_output = len(self.outputs)
        self.edges = edges
        self.n_edge = len(edges)
        # IDs of edges connected to nodes
        # ff_edges: outgoing edges, bp_edges: incoming edges
        self.ff_edges = [[j for j in range(self.n_edge) \
                if self.edges[j][0] == i] for i in xrange(self.n_node)]
        self.bp_edges = [[j for j in range(self.n_edge) \
                if self.edges[j][1] == i] for i in xrange(self.n_node)]

    def _init_training(self, **option):
        _option = dict(_default_option)
        for key in option:
            _option[key] = option[key]
        option = _option
        if option['reset']:
            self.W = [option['initW']*np.random.randn(self.nodes[x[0]].dim,
                self.nodes[x[1]].dim) for x in self.edges]
            self.b = [np.zeros((x.dim, )) for x in self.nodes]
        else:
            # if the network is to be trained using dropout, scale it
            if option['dropout'] > 0.:
                p = 1. - option['dropout']
                for i in xrange(self.n_node):
                    if i not in self.inputs and i not in self.outputs:
                        for e in self.ff_edges[i]:
                            self.W[e] /= p
        # squared error is the default cost function
        if not option['cost']:
            option['cost'] = [cost.squared_error]*self.n_output
        # RMSE is the default performance measure
        if not option['measure']:
            option['measure'] = [measure.rmse]*self.n_output
        N = option['batch_size']
        # temporary variables used for training
        # activations, gradients, parameters
        self._g = {
            'y': [gnp.zeros((N, x.dim)) for x in self.nodes],
            'dy': [gnp.zeros((N, x.dim)) for x in self.nodes],
            'W': [gnp.garray(x) for x in self.W],
            'dW': [gnp.zeros(x.shape) for x in self.W],
            'gW': [gnp.zeros(x.shape) for x in self.W],
            'b': [gnp.garray(x) for x in self.b],
            'db': [gnp.zeros(x.shape) for x in self.b],
            'gb': [gnp.zeros(x.shape) for x in self.b],
            }
        # temporary mask variables for dropout
        if option['dropout'] > 0.:
            self._g['mask'] = [gnp.zeros((N, x.dim)) for x in self.nodes]
        if option['cost_scale'] != None:
            self._g['cost_scale'] = [gnp.garray(x) \
                    for x in option['cost_scale']]
        self._g = EasyDict(self._g)
        return option

    # store the trained parameters and remove GPU variables
    def _finish_training(self, **option):
        self._copy_param(**option)
        del self._g
        gnp.free_reuse_cache()
        reload(gnp)
        gnp._restart_gpu()


    # copy parameters stored in GPU to host memory
    def _copy_param(self, **option):
        self.W = [x.asarray() for x in self._g.W]
        self.b = [x.asarray() for x in self._g.b]
        # calculate "mean weights" of dropout nets
        if option['dropout'] > 0.:
            p = 1. - option['dropout']
            for i in xrange(self.n_node):
                if i not in self.inputs and i not in self.outputs:
                    for e in self.ff_edges[i]:
                        self.W[e] *= p

    def fit(self, input_data, output_data, **option):
        for i_epoch in self.fit_iter(input_data, output_data, **option):
            pass

    def fit_iter(self, input_data, output_data, **option):
        # for easier use and backward compatibility,
        # convert list(np.ndarray) to list(Data)
        if type(input_data[0]) != Data:
            input_data = [Data(x) for x in input_data]
        if type(output_data[0]) != Data:
            output_data = [Data(x) for x in output_data]
        option = dict(self._init_training(**option))
        momentum_f = option['momentum']
        measures = option['measure']
        print option
        for i_epoch in xrange(option['n_epoch']):
            option['eps'] *= 0.998
            if i_epoch < 500:
                r_epoch = i_epoch/500.
                option['momentum'] = r_epoch*momentum_f + (1. - r_epoch)*0.5
            else:
                momentum = momentum_f
            for i in xrange(len(input_data[0])):
                inputs = [x[i] for x in input_data]
                outputs = [x[i] for x in output_data]
                self.fit_epoch(inputs, outputs, **option)
                self._copy_param(**option)
            if option['display'] and (i_epoch + 1) % 10 == 0:
                print 'epoch %d:' % (i_epoch + 1),
                print 'eps=%g, momentum=%g,' % \
                        (option['eps'], option['momentum'])
                measure_msg = self.evaluate(inputs, outputs, measures)
                print 'RMSE= %s' % measure_msg
            yield i_epoch
        self._finish_training(**option)

    def fit_epoch(self, inputs, outputs, **option):
        N = inputs[0].shape[0]
        chunk_size = min(N, option['chunk_size'])
        # augment dataset if the user chose to randomly drop views
        if option['drop_view'] and self.n_input > 1:
            N = inputs[0].shape[0]
            for k in xrange(self.n_input):
                padding_front = np.zeros((N*k, inputs[k].shape[1]))
                padding_back = np.zeros((N*(self.n_input - k - 1),\
                        inputs[k].shape[1]))
                inputs[k] = np.vstack([np.array(inputs[k]), padding_front, \
                        np.array(inputs[k]), padding_back])
            for k in xrange(self.n_output):
                outputs[k] = np.vstack([np.array(outputs[k]) \
                        for i in xrange(self.n_input + 1)])
        batches = Batch(inputs + outputs, chunk_size)
        batches.shuffle()
        n_batch = int(chunk_size/option['batch_size'])
        for batch in batches:
            g_inputs = [gnp.garray(x) for x in batch[:len(inputs)]]
            g_outputs = [gnp.garray(x) for x in batch[len(inputs):]]
            # add pre-specified noise pattern to input samples
            if option['noise']:
                for i in xrange(self.n_input):
                    noise_type = option['noise'][i]
                    param = option['noise_param'][i]
                    if noise_type == 'pepper':
                        mask = gnp.rand(*g_inputs[i].shape) > param
                        g_inputs[i] *= mask
                    elif noise_type == 'gaussian':
                        g_inputs[i] += param*gnp.randn(*g_inputs[i].shape)
                    else:
                        raise
            # mini-batch training
            for i_batch in xrange(n_batch):
                i1 = i_batch*option['batch_size']
                i2 = i1 + option['batch_size']
                g_batch_inputs = [x[i1:i2] for x in g_inputs]
                g_batch_outputs = [x[i1:i2] for x in g_outputs]
                self.fit_step(g_batch_inputs, g_batch_outputs, **option)

    # calculate the gradient using BP and update parameters
    def fit_step(self, inputs, outputs, **option):
        self._feed_forward(inputs, **option)
        self._back_propagate(outputs, **option)
        self._update(**option)

    def _feed_forward(self, inputs, **option):
        for i in xrange(self.n_node):
            if i not in self.inputs:
                self._g.y[i] *= 0
        for i in xrange(self.n_input):
            self._g.y[self.inputs[i]] = inputs[i]
        queue = list(self.inputs)
        check_list = [len(x) for x in self.bp_edges]
        while queue != []:
            i1 = queue.pop(0)
            if check_list[i1] == 0:
                if i1 not in self.inputs:
                    self._g.y[i1] = self.nodes[i1].f(self._g.y[i1]
                            + self._g.b[i1])
                    # dropout: randomly drop hidden nodes using binary masks
                    if option['dropout'] > 0. and i1 not in self.outputs:
                        self._g.mask[i1] = gnp.rand(*self._g.y[i1].shape) >\
                                option['dropout']
                        self._g.y[i1] *= self._g.mask[i1]
            for idx in self.ff_edges[i1]:
                i2 = self.edges[idx][1]
                if check_list[i1] == 0:
                    self._g.y[i2] += gnp.dot(self._g.y[i1], self._g.W[idx])
                    check_list[i2] -= 1
                    if i2 not in queue:
                        queue += [i2]
            if check_list[i1] != 0:
                queue += [i1]

    def _back_propagate(self, outputs, **option):
        N = outputs[0].shape[0]
        for i in xrange(self.n_node):
            if i not in self.outputs and i not in self.inputs:
                self._g.dy[i] *= 0
        for i in xrange(self.n_output):
            # apply pre-specified functions to the output
            i1 = self.outputs[i]
            self._g.dy[i1] = option['cost'][i](self._g.y[i1], outputs[i],\
                    is_d=True)
            # multiply a pre-specified scale to the gradient of errors
            if option['cost_scale'] != None:
                self._g.dy[i1] *= self._g.cost_scale[i]
        queue = list(self.outputs)
        check_list = [len(x) for x in self.ff_edges]
        while queue != []:
            i1 = queue.pop(0)
            if check_list[i1] == 0:
                if i1 not in self.inputs:
                    self._g.dy[i1] *= self.nodes[i1].df(self._g.y[i1])
                    self._g.gb[i1] = self._g.dy[i1].mean(0)
                    # dropout: randomly drop hidden nodes using binary masks
                    if option['dropout'] > 0. and i1 not in self.outputs:
                        self._g.dy[i1] *= self._g.mask[i1]
            for idx in self.bp_edges[i1]:
                i2 = self.edges[idx][0]
                if check_list[i1] == 0:
                    if i2 not in self.inputs:
                        self._g.dy[i2] += gnp.dot(self._g.dy[i1],
                                self._g.W[idx].T)
                    self._g.gW[idx] = gnp.dot(self._g.y[i2].T,
                            self._g.dy[i1])/N
                    check_list[i2] -= 1
                    if i2 not in queue:
                        queue += [i2]
            if check_list[i1] != 0:
                queue += [i1]

    def _update(self, **option):
        momentum = option['momentum']
        eps = option['eps']*(1. - momentum)
        # consider tied weights pairs!
        skip_list = [i2 for (_, i2) in option['tied']]
        skip_list += [i2 for (_, i2) in option['tied_t']]
        for (i1, i2) in option['tied_t']:
            self._g.gW[i1] += self._g.gW[i2].T
        for (i1, i2) in option['tied']:
            self._g.gW[i1] += self._g.gW[i2]
        # SGD with momentum
        for i in xrange(len(self._g.W)):
            if i not in skip_list:
                self._g.dW[i] *= momentum
                self._g.dW[i] += eps*self._g.gW[i]
                self._g.W[i] += self._g.dW[i]
        for i in xrange(len(self._g.b)):
            if i not in self.inputs:
                self._g.db[i] *= momentum
                self._g.db[i] += eps*self._g.gb[i]
                self._g.b[i] += self._g.db[i]
        # constrain squared norms of weights
        if option['w_norm']:
            w_norm = option['w_norm']
            for i in xrange(self.n_node):
                if i not in self.inputs:
                    w_ratio = self._g.b[i]**2
                    for e in self.bp_edges[i]:
                        w_ratio += (self._g.W[e]**2).sum(0)
                    # add 0.1 to prevent division by 0
                    w_ratio = w_norm/(w_ratio + 0.1)
                    w_ratio = w_ratio*(w_ratio < 1.) + (w_ratio >= 1.)
                    w_ratio = gnp.sqrt(w_ratio)
                    self._g.b[i] *= w_ratio
                    for e in self.bp_edges[i]:
                        self._g.W[e] *= w_ratio
        # make sure the shared weights share the same value
        for (i1, i2) in option['tied_t']:
            self._g.W[i2] = self._g.W[i1].T
        for (i1, i2) in option['tied']:
            self._g.W[i2] = self._g.W[i1]

    def feed_forward(self, inputs, **option):
        outputs = option['output'] if 'output' in option else self.outputs
        drop_list = option['drop'] if 'drop' in option else []
        N = inputs[0].shape[0]
        y = [np.zeros((N, x.dim)) for x in self.nodes]
        for i in xrange(self.n_input):
            y[self.inputs[i]] = inputs[i]
        queue = list(self.inputs)
        check_list = [len(x) for x in self.bp_edges]
        while queue != []:
            i1 = queue.pop(0)
            if i1 in drop_list:
                y[i1] *= 0
                check_list[i1] = 0
                for idx in self.ff_edges[i1]:
                    i2 = self.edges[idx][1]
                    check_list[i2] -= 1
                continue
            if check_list[i1] == 0:
                y[i1] = self.nodes[i1].f(y[i1] + self.b[i1])
            for idx in self.ff_edges[i1]:
                i2 = self.edges[idx][1]
                if check_list[i1] == 0:
                    y[i2] += np.dot(y[i1], self.W[idx])
                    check_list[i2] -= 1
                    if i2 not in queue:
                        queue += [i2]
            if check_list[i1] != 0:
                queue += [i1]
        return [y[i] for i in outputs]

    # evaluate the network with the given measure functions
    def evaluate(self, inputs, outputs, measures):
        predicted = self.feed_forward(inputs)
        K = len(outputs)
        return [measures[i](predicted[i], outputs[i]) for i in xrange(K)]
