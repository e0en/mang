from .edge import Edge
from .edge_ref import EdgeRef
from .max_pooling_edge import MaxPoolingEdge
from .convolutional_edge import ConvolutionalEdge


edge_table = {
    "full": Edge,
    "ref": EdgeRef,
    "max_pool": MaxPoolingEdge,
    "conv": ConvolutionalEdge,
    }
