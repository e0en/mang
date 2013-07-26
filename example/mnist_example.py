import sys
sys.path.append('../')

import numpy as np
import cudamat.gnumpy as gnp

from mang.feedforwardnetwork import FeedForwardNetwork as FF
import mang.node as mnode
import mang.visualization as vis


# download mnist test dataset from remote server
mnist_url = 'http://db.tt/D5E3GTvR'

mnist = {
        'X': np.random.randn(10000, 28*28),
        'label': np.random.rand(10000, 10) > 0.1,
        }

# create a simple neural network
nodes = [
        mnode.InputNode(28*28),
        mnode.ReLUNode(1024),
        mnode.SoftmaxNode(10, is_output=True),
        ]
edges = [(0, 1), (1, 2)]

nn = FF(nodes, edges)

# train the neural network
option = {
        'n_epoch': 100,
        'eps': 0.1,
        }
nn.fit([mnist['X']], [mnist['label']], **option)

# visualize the trained weights
vis.matrix_image(nn.W[0].T, shape=(28, 28)).save('W.png')
