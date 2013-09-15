import os
import unittest

import requests
import numpy as np

from mang.feedforwardnetwork import FeedForwardNetwork as FF
from mang import preprocessing as prep
from mang import measure


class TestFeedForwardNetwork(unittest.TestCase):
    """Test cases for FeedForwardNetwork class."""

    def setUp(self):
        """Prepare MNIST dataset for training."""

        # download mnist test dataset from remote server
        script_dir = os.path.dirname(__file__)
        mnist_path = os.path.join(script_dir, 'mnist.npz')
        if not os.path.exists(mnist_path):
            print "downloading mnist test dataset..."
            mnist_url = 'http://db.tt/D5E3GTvR'
            r = requests.get(mnist_url)
            fp = open(mnist_path, 'w')
            fp.write(r.content)
            fp.close()
            print "done!"

        data = np.load(mnist_path)
        self.data = {
            'input': prep.zca_whiten(data['X'])[0],
            'output': np.array(data['label']),
            }

        self.nodes = {
            'input': {
                'type': 'affine', 'shape': (28, 28, 1), 'use_bias': False, },
            'conv1': {
                'type': 'relu', 'shape': (24, 24, 16), 'shared': True, },
            'pool1': {
                'type': 'rnorm', 'shape': (12, 12, 16), 'norm_size': 2,
                'add_scale': 1., 'pow_scale': .75, },
            'conv2': {'type': 'relu', 'shape': (8, 8, 64), 'shared': True, },
            'pool2': {
                'type': 'rnorm', 'shape': (4, 4, 64), 'norm_size': 2,
                'add_scale': 1., 'pow_scale': .75, },
            'hidden': {'type': 'relu', 'shape': (500, )},
            'output': {'type': 'softmax', 'shape': (10, )},
            }

        self.edges = {
            ('input', 'conv1'): {
                'type': 'conv', 'filter_size': 5, 'padding': 0, 'stride': 1,
                },
            ('conv1', 'pool1'): {'type': 'max_pool', 'ratio': 2, },
            ('pool1', 'conv2'): {
                'type': 'conv', 'filter_size': 5, 'padding': 0, 'stride': 1,
                },
            ('conv2', 'pool2'): {'type': 'max_pool', 'ratio': 2, },
            ('pool2', 'hidden'): {'type': 'full'},
            ('hidden', 'output'): {'type': 'full'},
            }

    def test_convnet(self):
        """Test if network is trained properly."""

        net = FF(self.nodes, self.edges)

        node_param = {
            'conv1': {'init_b': 0., 'eps': 1e-1, },
            'conv2': {'init_b': 0., 'eps': 1e-1, },
            'hidden': {'init_b': 0., 'eps': 1e-1, },
            'output': {'cost': 'squared_error', },
            }
        edge_param = {
            ('input', 'conv1'): {'eps': 1e-1, },
            ('pool1', 'conv2'): {'eps': 1e-1, },
            ('pool2', 'hidden'): {'eps': 1e-1, },
            ('hidden', 'output'): {'eps': 1e-1, },
            }
        net.fit(self.data, n_epoch=10, batch_size=128,
                node_param=node_param, edge_param=edge_param)
        result = net.evaluate(self.data, {'output': 'accuracy', })
        self.assertTrue(result['output'] > 0.9)

    def test_save_load(self):
        """Test save/load functionality."""

        net = FF(self.nodes, self.edges)
        net.fit(self.data, n_epoch=2)
        output = net.feed_forward(self.data)['output']

        net.save('tmp.msgpack')
        net_recovered = FF.load('tmp.msgpack')
        os.system('rm tmp.msgpack')
        output_recovered = net_recovered.feed_forward(self.data)['output']

        conn_list = [
            ('input', 'conv1'),
            ('pool2', 'hidden'),
            ('hidden', 'output'),
            ]
        for conn in conn_list:
            w_original = net.edges[conn].W
            w_recovered = net_recovered.edges[conn].W
            self.assertTrue(np.allclose(w_original, w_recovered))

        for name in ['conv1', 'conv2', 'hidden', 'output']:
            b_original = net.nodes[name].b
            b_recovered = net_recovered.nodes[name].b
            self.assertTrue(np.allclose(w_original, w_recovered))
            self.assertTrue(np.allclose(b_original, b_recovered))

        print abs(output - output_recovered).mean()
        self.assertTrue(np.allclose(output, output_recovered))

    def test_shared_edges(self):
        """Test shared edges."""

        nodes = {
            'input': {'type': 'affine', 'shape': (28, 28), },
            'hidden': {'type': 'relu', 'shape': (500, ), },
            'output': {'type': 'affine', 'shape': (28, 28), },
            }
        edges = {
            ('input', 'hidden'): {'type': 'full', },
            ('hidden', 'output'): {
                'type': 'ref',
                'original': ('input', 'hidden'),
                'transpose': True,
                },
            }
        node_param = {
            'input': {'noise': 'pepper', 'noise_param': .2, }
            }

        net = FF(nodes, edges)

        data = {'input': self.data['input'], 'output': self.data['input'], }
        net.fit(data, n_epoch=2, node_param=node_param)
        net.feed_forward(data)

        net.save('tmp.msgpack')
        net_recovered = FF.load('tmp.msgpack')
        os.system('rm tmp.msgpack')
        net_recovered.feed_forward(data)


if __name__ == '__main__':
    unittest.main()
