"""Utils for testing compute engine ops."""
import tensorflow as tf

from distutils.version import LooseVersion


def tf_2_or_newer():
    return LooseVersion(tf.__version__) >= LooseVersion("2.0")


def eval_op(op):
    return tf.keras.backend.get_value(op)
