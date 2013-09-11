from .node import Node
from .softmax_node import SoftmaxNode
from .logistic_node import LogisticNode
from .tanh_node import TanhNode
from .relu_node import ReLUNode
from .response_normalization_node import ResponseNormalizationNode


_NODE_TYPE_LIST = [
    Node,
    TanhNode,
    LogisticNode,
    SoftmaxNode,
    ReLUNode,
    ResponseNormalizationNode
    ]
NODE_TABLE = {}

for node_type in _NODE_TYPE_LIST:
    NODE_TABLE[node_type._name] = node_type
