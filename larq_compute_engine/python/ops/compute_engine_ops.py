"""Use larq compute engine ops in python."""

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

_ops_lib = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_larq_compute_engine_ops.so")
)

bsign = _ops_lib.bsign

# binary convolution ops with the naming format bconv2d{bitpacking-bitwidth}
# the default bitpacking bitwidth is 64
bconv2d8 = _ops_lib.bconv2d8
bconv2d32 = _ops_lib.bconv2d32
bconv2d64 = _ops_lib.bconv2d64
bconv2d = _ops_lib.bconv2d64
