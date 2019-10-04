"""Use larq compute engine ops in python."""

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

_ops_lib = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_larq_compute_engine_ops.so")
)

bgemm = _ops_lib.bgemm
bsign = _ops_lib.bsign
