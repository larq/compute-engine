"""Use larq compute engine ops in python."""

import tensorflow as tf
from tensorflow.python.platform import resource_loader

compute_engine_ops = tf.load_op_library(
    resource_loader.get_path_to_datafile("_larq_compute_engine_ops.so")
)
bgemm = compute_engine_ops.bgemm
bsign = compute_engine_ops.bsign
