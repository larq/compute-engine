"""Tests for compute engine ops."""
import numpy as np
import tensorflow as tf
import sys
import pytest


try:
    from larq_compute_engine.python.ops.compute_engine_ops import (
        bconv2d8,
        bconv2d32,
        bconv2d64,
    )
    from larq_compute_engine.python.utils import eval_op, TestCase
except ImportError:
    from compute_engine_ops import bconv2d8, bconv2d32, bconv2d64
    from ..utils import eval_op, TestCase


class BConv2DTest(TestCase):
    @pytest.mark.parameterize("bconv_op", [bconv2d8, bconv2d32, bconv2d64])
    @pytest.mark.parameterize("data_type", [np.float32, np.float64])
    @pytest.mark.parameterize("data_format", ["NHWC"])
    @pytest.mark.parameterize("in_size", [[10, 10], [11, 11], [10, 11]])
    @pytest.mark.parameterize("filter_size", [[3, 3], [4, 5]])
    @pytest.mark.parameterize("in_channel", [1, 31, 32, 33, 64])
    @pytest.mark.parameterize("out_channel", [1, 16])
    @pytest.mark.parameterize("hw_stride", [[1, 1], [2, 2]])
    @pytest.mark.parameterize("padding", ["VALID", "SAME"])
    def test_bconv(
        self,
        bconv_op,
        dtype,
        data_format,
        in_size,
        filter_size,
        in_channel,
        out_channel,
        hw_stride,
        padding,
    ):
        batch_size = out_channel
        h, w = in_size
        fh, fw = filter_size
        if data_format == "NHWC":
            ishape = [batch_size, h, w, in_channel]
            strides = [1] + hw_stride + [1]
        else:
            raise ValueError("Unknown data_format: " + str(data_format))
        fshape = [fh, fw, in_channel, out_channel]

        sample_list = [-1, 1]
        inp = np.random.choice(sample_list, np.prod(ishape)).astype(dtype)
        inp = np.reshape(inp, ishape)

        filt = np.random.choice(sample_list, np.prod(fshape)).astype(dtype)
        filt = np.reshape(filt, fshape)

        with self.test_session():
            output = eval_op(
                bconv_op(inp, filt, strides, padding, data_format=data_format)
            )
            expected = eval_op(
                tf.nn.conv2d(inp, filt, strides, padding, data_format=data_format)
            )
            self.assertAllClose(output, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
