import os

import requests
import numpy as np

from mang.feedforwardnetwork import FeedForwardNetwork as FF
from e0en import preprocess as prep
from mang.visualization import filter_image


# download mnist test dataset from remote server
if not os.path.exists('mnist.npz'):
    print 'downloading mnist test dataset...'
    mnist_url = 'http://db.tt/D5E3GTvR'
    r = requests.get(mnist_url)
    fp = open('mnist.npz', 'w')
    fp.write(r.content)
    fp.close()
    print 'done!'

data = np.load('mnist.npz')
mnist = {
    "input": prep.ZCAwhiten(data["X"])[0],
    "output": np.array(data["label"]),
    }
del data
print "data is loaded"
# create a simple, convolutional neural network
nodes = {
    "input": {
        "type": "affine",
        "shape": (28, 28, 1),
        "use_bias": False,
        },
    "conv1": {
        "type": "relu",
        "shape": (24, 24, 16),
        "shared": True,
        },
    "pool1": {
        "type": "rnorm",
        "shape": (12, 12, 16),
        "use_bias": False,
        "norm_size": 2,
        "add_scale": 1.,
        "pow_scale": .75,
        },
    "conv2": {
        "type": "relu",
        "shape": (8, 8, 64),
        "shared": True,
        },
    "pool2": {
        "type": "rnorm",
        "shape": (4, 4, 64),
        "use_bias": False,
        "norm_size": 2,
        "add_scale": 1.,
        "pow_scale": .75,
        },
    "hidden": {"type": "relu", "shape": (500, )},
    "output": {"type": "softmax", "shape": (10, )},
    }

edges = {
    ("input", "conv1"): {
        "type": "conv",
        "filter_size": 5,
        "padding": 0,
        "stride": 1,
        },
    ("conv1", "pool1"): {
        "type": "max_pool",
        "ratio": 2,
        },
    ("pool1", "conv2"): {
        "type": "conv",
        "filter_size": 5,
        "padding": 0,
        "stride": 1,
        },
    ("conv2", "pool2"): {
        "type": "max_pool",
        "ratio": 2,
        },
    ("pool2", "hidden"): {"type": "full"},
    ("hidden", "output"): {"type": "full"},
    }

nn = FF(nodes, edges)


# train the neural network
def callback_function(nn, stat):
    i_epoch = stat[-1]["epoch"]

    mat = nn.nodes["input"].y.asarray()
    img = filter_image(mat, shape=nn.nodes["input"].shape)
    img.save("mnist_input_%d.png" % i_epoch)

    mat = nn.nodes["hidden"].y.asarray()
    img = filter_image(mat, shape=(25, 20))
    img.save("mnist_hidden_%d.png" % i_epoch)

    mat = nn.nodes["output"].y.asarray()
    img = filter_image(mat, shape=(2, 5))
    img.save("mnist_output_%d.png" % i_epoch)

    mat = nn.nodes["output"].dy.asarray()
    img = filter_image(mat, shape=(2, 5))
    img.save("mnist_dy_%d.png" % i_epoch)

    mat = nn.edges["input", "conv1"].W.asarray()
    filter_size = nn.edges["input", "conv1"].filter_size
    n_channel = nn.nodes["input"].shape[-1]
    img = filter_image(mat, shape=(filter_size, filter_size, n_channel))
    img.save("mnist_Wconv1_%d.png" % i_epoch)

    mat = nn.edges["pool1", "conv2"].W.asarray()
    filter_size = nn.edges["pool1", "conv2"].filter_size
    n_channel = nn.nodes["pool1"].shape[-1]
    img = filter_image(mat, shape=(filter_size, filter_size, n_channel))
    img.save("mnist_Wconv2_%d.png" % i_epoch)

    print "epoch %d finished. testing..." % i_epoch
    print i_epoch, nn.evaluate(mnist, {"output": "accuracy"})


option = {
    'n_epoch': 100,
    'batch_size': 128,
    "edge_param": {
        ("input", "conv1"): {
            "eps": 1e-1,
            },
        ("pool1", "conv2"): {
            "eps": 1e-1,
            },
        ("pool2", "hidden"): {
            "eps": 1e-1,
            },
        ("hidden", "output"): {
            "eps": 1e-1,
            },
        },
    "node_param": {
        "conv1": {
            "init_b": 0.,
            "eps": 1e-1,
            },
        "conv2": {
            "init_b": 0.,
            "eps": 1e-1,
            },
        "hidden": {
            "init_b": 0.,
            "eps": 1e-1,
            },
        "output": {
            "cost": "squared_error",
            },
        },
    "callback": callback_function,
    }
nn.fit(mnist, **option)
