"""Utils for testing compute engine ops."""
import tensorflow as tf

from distutils.version import LooseVersion


def tf_2_or_newer():
    return LooseVersion(tf.__version__) >= LooseVersion("2.0")


def eval_op(op):
    if tf_2_or_newer():
        return op  # op.numpy() also works
    else:
        return op.eval()


class TestCase(tf.test.TestCase):
    """A test case that can be run with pytest parameterized tests"""

    def run(self, **kargs):
        pass
