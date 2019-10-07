"""Utils for testing compute engine ops."""
from tensorflow import __version__
from distutils.version import LooseVersion


def tf_2_or_newer():
    return LooseVersion(__version__) >= LooseVersion("2.0")


def eval_op(op):
    if tf_2_or_newer():
        return op  # op.numpy() also works
    else:
        return op.eval()
