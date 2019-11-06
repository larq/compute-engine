"""Tests for compute engine ops."""
import numpy as np
import tensorflow as tf
from absl.testing import parameterized


try:
    from larq_compute_engine.python.ops.compute_engine_ops import bsign
    from larq_compute_engine.python.utils import eval_op
except ImportError:
    from compute_engine_ops import bsign
    from compute_engine_ops.python.utils import eval_op


class SignTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(np.int8, np.int32, np.int64)
    def test_sign_int(self, dtype):
        with self.test_session():
            x = np.array([[2, -5], [-3, 0]]).astype(dtype)
            expected_output = np.array([[1, -1], [-1, 1]])
            self.assertAllClose(eval_op(bsign(x)), expected_output)

    # Test for +0 and -0 floating points.
    # We have sign(+0) = 1 and sign(-0) = -1
    @parameterized.parameters(np.float32, np.float64)
    def test_sign_float(self, dtype):
        with self.test_session():
            x = np.array([[0.1, -5.8], [-3.0, 0.00], [0.0, -0.0]]).astype(dtype)
            expected_output = np.array([[1, -1], [-1, 1], [1, -1]])
            self.assertAllClose(eval_op(bsign(x)), expected_output)


if __name__ == "__main__":
    tf.test.main()
