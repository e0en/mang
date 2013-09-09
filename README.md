mang
====

Mang is a neural network library for python 2.x.
It requires the packages below:

* numpy (>= 1.6.2)
* cudamat (https://code.google.com/p/cudamat/)

Please note that Mang uses a very slightly customized version of cudamat.

A Simple Example
----------------
Mang defines a feed-forward network by specifying nodes and edges.
Input nodes and output nodes are determined by the topology of graph,
so that one does not need to explicitely specify input and output nodes.

    from mang.feedforwardnetwork import FeedForwardNetwork
    
    
    nodes = {
        "input": {"type": "affine", "shape": (28, 28), },
        "hidden": {"type": "relu", "shape": (1024, ), },
        "output": {"type": "relu", "shape": (10, ), },
        }
    edges = {
        ("input", "hidden"): {"type": "full"},
        ("hidden", "output"): {"type": "full"},
        }
    nn = FeedForwardNetwork(nodes, edges)

Training a network is also easy. Training data must be given as a dictionary
of numpy arrays, whose keys corresponds to node names of the network.

    data = {
        "input": mnist["X"],
        "output": mnist["label"],
        }
    nn.fit(data, n_epoch=100, cost=cost)

Trained networks can be tested by calculating their activations or directly
evaluating it on test datasets.

    data_test = {"input": mnist["X_test"],}
    result = nn.feed_forward(data_test)
    predicted_labels = output["output"]
    
    data_test = {"input": mnist["X_test"], "output": mnist["label_test"],}
    measures = {"output": "accuracy", }
    performance = nn.evaluate(data_test, measure=measures)
