import os

import requests
import numpy as np
import cudamat.gnumpy as gnp

from mang.feedforwardnetwork import FeedForwardNetwork as FF
from mang import cost
from mang import measure
from mang import node as mnode
from mang import visualization as vis


# download mnist test dataset from remote server
if not os.path.exists('mnist.npz'):
    print 'downloading mnist test dataset...'
    mnist_url = 'http://db.tt/D5E3GTvR'
    r = requests.get(mnist_url)
    fp = open('mnist.npz', 'w')
    fp.write(r.content)
    fp.close()
    print 'done!'

mnist = np.load('mnist.npz')

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
        'eps': 0.01,
        'cost': [cost.cross_entropy],
        'measure': [measure.accuracy],
        }
nn.fit([mnist['X'] - mnist['X'].mean(0)], [mnist['label']], **option)

# visualize the trained weights
vis.matrix_image(nn.W[0].T, shape=(28, 28)).save('W.png')
