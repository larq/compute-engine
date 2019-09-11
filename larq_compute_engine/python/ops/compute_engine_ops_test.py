"""Tests for compute engine ops."""
import numpy as np
import tensorflow as tf

try:
    from larq_compute_engine.python.ops.compute_engine_ops import bgemm, fast_sign
    print("Imported larq compute engine from pip package")
except ImportError:
    from compute_engine_ops import bgemm, fast_sign
    print("Imported larq compute engine from local file")


class BGEMMTest(tf.test.TestCase):
    def test_bgemm_int32(self):
        with self.test_session():
            input_a = np.array([[1, 1], [1, 1]]).astype(np.int32)
            input_b = np.array([[1, 1], [1, 1]]).astype(np.int32)
            expected_output = np.array([[0, 0], [0, 0]])

            self.assertAllClose(bgemm(input_a, input_b).eval(), expected_output)

class SignTest(tf.test.TestCase):
    def test_sign_int8(self):
        with self.test_session():
            x = np.array([[2, -5], [-3, 0]]).astype(np.int8)
            expected_output = np.array([[1, -1], [-1, 1]])
            self.assertAllClose(fast_sign(x).eval(), expected_output)

    def test_sign_int32(self):
        with self.test_session():
            x = np.array([[2, -5], [-3, 0]]).astype(np.int32)
            expected_output = np.array([[1, -1], [-1, 1]])
            self.assertAllClose(fast_sign(x).eval(), expected_output)

    def test_sign_float32(self):
        with self.test_session():
            # Test for +0 and -0 floating points.
            # We have sign(+0) = 1 and sign(-0) = -1
            x = np.array([[0.1, -5.8], [-3.0, 0.00], [0.0, -0.0]]).astype(np.float32)
            expected_output = np.array([[1, -1], [-1, 1], [1, -1]]).astype(np.float32)
            self.assertAllClose(fast_sign(x).eval(), expected_output)

