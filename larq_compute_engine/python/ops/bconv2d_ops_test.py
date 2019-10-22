"""Tests for compute engine ops."""
import numpy as np
import tensorflow as tf
import itertools


try:
    from larq_compute_engine.python.ops.compute_engine_ops import (
        bconv2d8,
        bconv2d32,
        bconv2d64,
    )
    from larq_compute_engine.python.utils import eval_op
except ImportError:
    from compute_engine_ops import bconv2d8, bconv2d32, bconv2d64
    from ..utils import eval_op


class BConv2DTest(tf.test.TestCase):
    def __test_bconv_no_padding(self, bconv_op):
        data_types = [np.float32, np.float64]
        data_formats = ["NHWC"]
        in_channels = list(range(1, 10)) + [31, 32, 33] + [63, 64, 65]
        out_channels = [1, 4, 16, 32]
        hw_strides = [[1, 1], [2, 2]]
        paddings = ["VALID", "SAME"]

        args_lists = [
            data_types,
            data_formats,
            in_channels,
            out_channels,
            hw_strides,
            paddings,
        ]
        for args in itertools.product(*args_lists):
            dtype, data_format, in_channel, out_channel, hw_stride, padding = args

            batch_size = out_channel
            h, w, fh, fw = 20, 20, 3, 3
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

    def test_bconv2d8_no_padding(self):
        self.__test_bconv_no_padding(bconv2d8)

    def test_bconv2d32_no_padding(self):
        self.__test_bconv_no_padding(bconv2d32)

    def test_bconv2d64_no_padding(self):
        self.__test_bconv_no_padding(bconv2d64)


if __name__ == "__main__":
    tf.test.main()
