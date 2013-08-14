import os

import requests
import numpy as np
import cudamat.gnumpy as gnp

from mang.feedforwardnetwork import FeedForwardNetwork as FF
from mang import cost
from mang import measure
from mang import node as mnode

import e0en.visualization as vis


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
        "output": {"type": "affine", "shape": (28, 28)},
        }
edges = {
        ("input", "hidden"): {"type": "full"},
        ("hidden", "output"): {"type": "full"},
        }
nn = FF(nodes, edges)

def callback_function(nn, i_epoch, data):
    print i_epoch, nn.evaluate(data, {"output": "mse"})

data = {
        "input": mnist['X'] - mnist['X'].mean(0),
        "output": mnist['X'] - mnist['X'].mean(0),
        }

option = {
        'n_epoch': 100,
        'eps': 0.1,
        'cost': {"output": "squared_error"},
        "measure": {"output": "accuracy"},
        "shared_t": {
            ("hidden", "output"): ("input", "hidden"),
            },
        "callback": lambda x, y: callback_function(x, y, data),
        }

nn.fit(data, **option)
