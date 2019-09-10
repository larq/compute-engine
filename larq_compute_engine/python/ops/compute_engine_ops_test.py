"""Tests for compute engine ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import test

try:
    from larq_compute_engine.python.ops.compute_engine_ops import bgemm
except ImportError:
    from compute_engine_ops import bgemm


class BGEMMTest(test.TestCase):

    def test_bgemm_int32(self):
        with self.test_session():
            input_a = np.array([[1, 1], [1, 1]]).astype(np.int32)
            input_b = np.array([[1, 1], [1, 1]]).astype(np.int32)
            expected_output = np.array([[0, 0], [0, 0]])

            self.assertAllClose(
                bgemm(input_a, input_b).eval(), expected_output)


if __name__ == '__main__':
    test.main()
