from .node import Node
from .softmax_node import SoftmaxNode
from .logistic_node import LogisticNode
from .tanh_node import TanhNode
from .relu_node import ReLUNode
from .response_normalization_node import ResponseNormalizationNode


node_table = {
    "affine": Node,
    "tanh": TanhNode,
    "logistic": LogisticNode,
    "softmax": SoftmaxNode,
    "relu": ReLUNode,
    "rnorm": ResponseNormalizationNode,
    }
