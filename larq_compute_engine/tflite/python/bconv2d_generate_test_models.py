"""Test for compute engine TF lite Python wrapper"""
import numpy as np
import tensorflow as tf
import larq_compute_engine as lqce
import os
import sys
import pytest

from tflite_runtime.interpreter import Interpreter


def _create_bconv2d_layer(
    bconv_op,
    in_channels,
    filters,
    kernel_size,
    hw_strides=(1, 1),
    padding="VALID",
    dilations=(1, 1),
):
    strides = [1] + list(hw_strides) + [1]
    dilations = [1] + list(dilations) + [1]
    fshape = [kernel_size, kernel_size, in_channels, filters]
    filt_tf = np.random.choice([-1, 1], fshape)
    filt_lite = np.copy(np.moveaxis(filt_tf, 3, 0))

    def _layer_tf(x):
        # Use native tf op because it supports dilations
        return tf.nn.conv2d(
            x, filt_tf, strides, padding, data_format="NHWC", dilations=dilations
        )

    def _layer_lite(x):
        return bconv_op(
            x,
            filt_lite,
            strides,
            padding,
            data_format="NHWC",
            filter_format="OHWI",
            dilations=dilations,
        )

    return _layer_tf, _layer_lite


def _create_combi_model(
    bconv_op, input_shape, filters, kernel_size, strides, padding, dilations
):
    layer_tf, layer_lite = _create_bconv2d_layer(
        bconv_op, input_shape[2], filters, kernel_size, strides, padding, dilations
    )

    img_input = tf.keras.layers.Input(shape=input_shape)

    out_tf = layer_tf(img_input)
    out_lite = layer_lite(img_input)

    model_combi = tf.keras.Model(inputs=img_input, outputs=[out_lite, out_tf])

    return model_combi


def invoke_inference(model, data, filename=None):
    # Convert to tflite
    converter = lqce.ModelConverter(model)
    tflite_model = converter.convert(filename)

    # Setup tflite
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_type = input_details[0]["dtype"]
    input_shape = input_details[0]["shape"]

    np.testing.assert_equal(input_shape, data.shape)
    np.testing.assert_equal(input_type, data.dtype)

    interpreter.set_tensor(input_details[0]["index"], data)

    # Run tflite
    interpreter.invoke()

    # return tflite result
    return [interpreter.get_tensor(detail["index"]) for detail in output_details]


@pytest.mark.parametrize("bconv_op", [lqce.bconv2d8, lqce.bconv2d32, lqce.bconv2d64])
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("in_channel", [1, 3])
@pytest.mark.parametrize("out_channel", [1, 4])
@pytest.mark.parametrize("input_size", [14])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("strides", [[1, 1], [2, 3]])
@pytest.mark.parametrize("padding", ["VALID", "SAME"])
@pytest.mark.parametrize("dilations", [[1, 1], [2, 3]])
def test_bconv2d_op_inference(
    bconv_op,
    data_type,
    in_channel,
    out_channel,
    input_size,
    kernel_size,
    strides,
    dilations,
    padding,
):
    input_shape = [input_size, input_size, in_channel]
    model_combi = _create_combi_model(
        bconv_op, input_shape, out_channel, kernel_size, strides, padding, dilations
    )

    if not os.path.exists("testing_models"):
        os.makedirs("testing_models")

    filename = f"testing_models/testnet_{bconv_op.__name__}_input_{input_size}x{input_size}x{in_channel}_kernel_{kernel_size}x{kernel_size}x{out_channel}_stride_{strides[0]}x{strides[1]}_padding_{padding}_dilations_{dilations[0]}x{dilations[1]}.tflite"

    input_shape = [1] + input_shape
    input_data = np.random.choice([-1, 1], input_shape).astype(data_type)

    combi_result = invoke_inference(model_combi, input_data, filename)

    np.testing.assert_allclose(combi_result[0], combi_result[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
