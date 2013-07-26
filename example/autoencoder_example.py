import os

import requests
import numpy as np
import cudamat.gnumpy as gnp

from mang.autoencoder import AutoEncoder as AE
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
hidden_nodes = [mnode.ReLUNode(500)]
output_nodes = [mnode.AffineNode(28*28)]
edges = [(0, 0)]
nn = AE(hidden_nodes, output_nodes, edges)

# train the neural network
option = {
        'n_epoch': 100,
        'eps': 0.1,
        'tied': True,
        }
nn.fit([mnist['X'] - mnist['X'].mean(0)], **option)

# visualize the trained weights
vis.matrix_image(nn.W[0].T, shape=(28, 28)).save('W.png')
