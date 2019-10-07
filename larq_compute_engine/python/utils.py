"""Utils for testing compute engine ops."""
from tensorflow import __version__
from distutils.version import LooseVersion


def eval_op(op):
    if LooseVersion(__version__) >= LooseVersion("2.0"):
        return op  # op.numpy() also works
    else:
        return op.eval()
