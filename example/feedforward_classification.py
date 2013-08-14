import os

import requests
import numpy as np
import cudamat.gnumpy as gnp

from mang.feedforwardnetwork import FeedForwardNetwork as FF
from mang import cost
from mang import measure
from mang import node as mnode


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
nodes = {
        "input": {"type": "affine", "shape": (28, 28)},
        "hidden": {"type": "relu", "shape": (1024, )},
        "output": {"type": "relu", "shape": (10, )},
        }
edges = {
        ("input", "hidden"): {"type": "full"},
        ("hidden", "output"): {"type": "full"},
        }
nn = FF(nodes, edges)

# train the neural network
def callback_function(nn, i_epoch):
    print i_epoch, nn.evaluate(data, {"output": "accuracy"})

option = {
        'n_epoch': 10,
        'eps': 0.1,
        'cost': {"output": "squared_error"},
        "measure": {"output": "accuracy"},
        "callback": callback_function,
        }
data = {
        "input": mnist['X'] - mnist['X'].mean(0),
        "output": mnist['label'],
        }
nn.fit(data, **option)
