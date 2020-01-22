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
    from larq_compute_engine.python.utils import eval_op
except ImportError:
    from compute_engine_ops import bconv2d8, bconv2d32, bconv2d64
    from ..utils import eval_op


@pytest.mark.parametrize("bconv_op", [bconv2d8, bconv2d32, bconv2d64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("data_format", ["NHWC"])
@pytest.mark.parametrize("in_size", [[10, 10], [10, 11]])
@pytest.mark.parametrize("filter_size", [[3, 3], [2, 3]])
@pytest.mark.parametrize("in_channel", [31, 32, 33])
@pytest.mark.parametrize("out_channel", [1, 16])
@pytest.mark.parametrize("strides", [[1, 1], [2, 3]])
@pytest.mark.parametrize("padding", ["VALID", "SAME"])
@pytest.mark.parametrize("dilations", [[1, 1]])  # Once we support dilations: [2, 3]
def test_bconv(
    bconv_op,
    dtype,
    data_format,
    in_size,
    filter_size,
    in_channel,
    out_channel,
    strides,
    padding,
    dilations,
):
    batch_size = out_channel
    h, w = in_size
    fh, fw = filter_size
    if data_format == "NHWC":
        ishape = [batch_size, h, w, in_channel]
        strides = [1] + strides + [1]
        dilations = [1] + dilations + [1]
    else:
        raise ValueError("Unknown data_format: " + str(data_format))
    fshape = [fh, fw, in_channel, out_channel]

    sample_list = [-1, 1]
    inp = np.random.choice(sample_list, np.prod(ishape)).astype(dtype)
    inp = np.reshape(inp, ishape)

    filt = np.random.choice(sample_list, np.prod(fshape)).astype(dtype)
    filt = np.reshape(filt, fshape)

    fused_multiply = np.full(shape=(out_channel), fill_value=-2)
    fused_add = np.full(shape=(out_channel), fill_value=fh * fw * in_channel)

    output = eval_op(
        bconv_op(
            inp,
            filt,
            fused_multiply,
            fused_add,
            strides,
            padding,
            dilations=dilations,
            data_format=data_format,
        )
    )
    expected = eval_op(
        tf.nn.conv2d(
            inp, filt, strides, padding, dilations=dilations, data_format=data_format
        )
    )
    np.testing.assert_allclose(output, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
