import cPickle as pickle

import numpy as np

from mang.feedforwardnetwork import FeedForwardNetwork as FF
from mang import visualization as vis
from e0en import preprocess as prep


def oneofk(label_list):
    label_names = list(set(label_list))
    label_names.sort()
    n_class = len(label_names)
    n = len(label_list)
    label = np.zeros((n, n_class))
    for i in xrange(n):
        idx = label_names.index(label_list[i])
        label[i, idx] = 1
    return label

# load cifar-10 dataset
images = np.zeros((0, 3072))
labels = np.zeros((0, 10))
for i in xrange(4):
    with open("cifar-10/data_batch_%d" % (i + 1), "rb") as fp:
        raw_data = pickle.load(fp)
        images = np.vstack((images, raw_data["data"]))
        labels = np.vstack((labels, oneofk(raw_data["labels"])))
cifar10 = {"input": images, "output": labels}
(cifar10["input"], stat) = prep.center(cifar10["input"])
scale = abs(cifar10["input"]).max()
cifar10["input"] /= scale

cifar10_valid = {}
with open("cifar-10/data_batch_5", "rb") as fp:
    raw_data = pickle.load(fp)
    cifar10_valid["input"] = np.array(raw_data["data"])
    cifar10_valid["output"] = oneofk(raw_data["labels"])

cifar10_valid["input"] = prep.center(cifar10_valid["input"], stat)
cifar10_valid["input"] /= scale

del raw_data

print "data is ready"

# create a simple, convolutional neural network
nodes = {
    "input": {"type": "affine", "shape": (32, 32, 3)},
    "conv1": {"type": "relu", "shape": (28, 28, 32), "shared": True},
    "pool1": {
        "type": "rnorm",
        "use_bias": False,
        "shape": (14, 14, 32),
        "norm_size": 2,
        "add_scale": 1.,
        "pow_scale": .75,
        },
    "conv2": {"type": "relu", "shape": (10, 10, 32), "shared": True},
    "pool2": {
        "type": "rnorm",
        "use_bias": False,
        "shape": (5, 5, 32),
        "norm_size": 2,
        "add_scale": 1.,
        "pow_scale": .75,
        },
    "conv3": {"type": "relu", "shape": (3, 3, 64), "shared": True},
    "hidden": {"type": "relu", "shape": (64, )},
    "output": {"type": "softmax", "shape": (10, ), },
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
    ("pool2", "conv3"): {
        "type": "conv",
        "filter_size": 3,
        "padding": 0,
        "stride": 1,
        },
    ("conv3", "hidden"): {"type": "full"},
    ("hidden", "output"): {"type": "full"},
    }

nn = FF(nodes, edges)


# train the neural network
def callback_function(nn, stat):
    i_epoch = stat[-1]["epoch"]
    mat = nn.edges["input", "conv1"].W.asarray()
    filter_size = nn.edges["input", "conv1"].filter_size
    n_channel = nn.nodes["input"].shape[-1]
    img = vis.filter_image(mat, shape=(filter_size, filter_size, n_channel))
    img.save("cifar10_Wconv1_%d.png" % i_epoch)

    for name in ["input", "conv1", "pool1", "conv2", "pool2", "conv3"]:
        mat = nn.nodes[name].y.asarray()
        img = vis.filter_image(mat, shape=nn.nodes[name].shape)
        img.save("cifar10_%s_%d.png" % (name, i_epoch))

    name = "hidden"
    mat = nn.nodes[name].y.asarray()
    img = vis.filter_image(mat, shape=(8, 8))
    img.save("cifar10_%s_%d.png" % (name, i_epoch))

    name = "output"
    mat = nn.nodes[name].y.asarray()
    img = vis.filter_image(mat, shape=(5, 2))
    img.save("cifar10_%s_%d.png" % (name, i_epoch))

    print "epoch %d finished. testing..." % i_epoch
    performance = nn.evaluate(cifar10, {"output": "accuracy"})
    print "* training accuracy: %g" % performance["output"]
    performance = nn.evaluate(cifar10_valid, {"output": "accuracy"})
    print "* validation accuracy: %g" % performance["output"]


option = {
    'n_epoch': 100,
    'batch_size': 64,
    "eps_decay": 0.95,
    "edge_param": {
        ("input", "conv1"): {
            "eps": 1e-1,
            "init_w": 1e-4,
            "momentum_i": 0.9,
            },
        ("pool1", "conv2"): {
            "eps": 1e-1,
            "momentum_i": 0.9,
            },
        ("pool2", "conv3"): {
            "eps": 1e-1,
            "momentum_i": 0.9,
            },
        ("conv3", "hidden"): {
            "eps": 1e-1,
            "momentum_i": 0.9,
            },
        ("hidden", "output"): {
            "init_w": 1e-1,
            },
        },
    "node_param": {
        "output": {
            "cost": "squared_error",
            "measure": "accuracy",
            },
        },
    "callback": callback_function,
    }
nn.fit(cifar10, **option)
