from .node import Node
from .softmax_node import SoftmaxNode
from .logistic_node import LogisticNode
from .tanh_node import TanhNode
from .relu_node import ReLUNode
'''
from .pooling_node import PoolingNode
from .contrast_normalization_node import ContrastNormalizationNode
'''


node_table = {
        "affine": Node,
        "tanh": TanhNode,
        "logistic": LogisticNode,
        "softmax": SoftmaxNode,
        "relu": ReLUNode,
        }
