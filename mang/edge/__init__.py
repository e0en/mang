from .edge import Edge
from .edge_ref import EdgeRef
from .max_pooling_edge import MaxPoolingEdge
from .convolutional_edge import ConvolutionalEdge
from .deconvolutional_edge import DeconvolutionalEdge

_EDGE_TYPE_LIST = [
    Edge,
    EdgeRef,
    MaxPoolingEdge,
    ConvolutionalEdge,
    DeconvolutionalEdge,
    ]

EDGE_TABLE = {}
for edge_type in _EDGE_TYPE_LIST:
    EDGE_TABLE[edge_type._name] = edge_type
