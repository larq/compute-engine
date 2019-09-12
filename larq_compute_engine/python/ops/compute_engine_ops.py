"""Use larq compute engine ops in python."""

import tensorflow as tf
import os


def get_path_to_datafile(path):
    """Get the path to the specified file in the data dependencies.
    Args:
    path: a string resource path relative to the current file
    Returns:
    The path to the specified data file
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root_dir, path)


_ops_lib = tf.load_op_library(get_path_to_datafile("_larq_compute_engine_ops.so"))
bgemm = _ops_lib.bgemm
bsign = _ops_lib.bsign
