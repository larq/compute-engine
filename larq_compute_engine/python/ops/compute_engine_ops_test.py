"""Tests for compute engine ops."""
import numpy as np
import tensorflow as tf
from distutils.version import LooseVersion


try:
    from larq_compute_engine.python.ops.compute_engine_ops import bgemm, bsign
except ImportError:
    from compute_engine_ops import bgemm, bsign


def get(x):
    if LooseVersion(tf.__version__) >= LooseVersion("2.0"):
        return x  # or x.numpy() also works
    else:
        return x.eval()


class BGEMMTest(tf.test.TestCase):
    def test_bgemm_int32(self):
        with self.test_session():
            input_a = np.array([[1, 1], [1, 1]]).astype(np.int32)
            input_b = np.array([[1, 1], [1, 1]]).astype(np.int32)
            expected_output = np.array([[0, 0], [0, 0]])

            self.assertAllClose(get(bgemm(input_a, input_b)), expected_output)


class SignTest(tf.test.TestCase):
    def run_test_for_integers(self, dtype):
        with self.test_session():
            x = np.array([[2, -5], [-3, 0]]).astype(dtype)
            expected_output = np.array([[1, -1], [-1, 1]])
            self.assertAllClose(get(bsign(x)), expected_output)

    # Test for +0 and -0 floating points.
    # We have sign(+0) = 1 and sign(-0) = -1
    def run_test_for_floating(self, dtype):
        with self.test_session():
            x = np.array([[0.1, -5.8], [-3.0, 0.00], [0.0, -0.0]]).astype(dtype)
            expected_output = np.array([[1, -1], [-1, 1], [1, -1]])
            self.assertAllClose(get(bsign(x)), expected_output)

    def test_sign_int8(self):
        self.run_test_for_integers(np.int8)

    def test_sign_int32(self):
        self.run_test_for_integers(np.int32)

    def test_sign_int64(self):
        self.run_test_for_integers(np.int64)

    def test_sign_float32(self):
        self.run_test_for_floating(np.float32)

    def test_sign_float64(self):
        self.run_test_for_floating(np.float64)


if __name__ == "__main__":
    tf.test.main()
