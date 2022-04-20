import sys

import larq as lq
import pytest
import tensorflow as tf

from larq_compute_engine.mlir.python.converter import convert_keras_model
from larq_compute_engine.mlir.python.util import strip_lcedequantize_ops


def toy_model_sign(**kwargs):
    img = tf.keras.layers.Input(shape=(224, 224, 3))
    x = lq.layers.QuantConv2D(
        256,
        kernel_size=3,
        strides=1,
        padding="same",
        pad_values=1,
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
    )(img)
    x = lq.quantizers.SteSign()(x)
    return tf.keras.Model(inputs=img, outputs=x)


def quant(x):
    return tf.quantization.fake_quant_with_min_max_vars(x, -3.0, 3.0)


def toy_model_int8_sign(**kwargs):
    img = tf.keras.layers.Input(shape=(224, 224, 3))
    x = quant(img)
    x = lq.layers.QuantConv2D(
        256,
        kernel_size=3,
        strides=1,
        padding="same",
        pad_values=1,
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
    )(x)
    x = lq.quantizers.SteSign()(x)
    x = quant(x)
    return tf.keras.Model(inputs=img, outputs=x)


@pytest.mark.parametrize("model_cls", [toy_model_sign, toy_model_int8_sign])
@pytest.mark.parametrize("inference_input_type", [tf.float32, tf.int8])
@pytest.mark.parametrize("inference_output_type", [tf.float32, tf.int8])
def test_strip_lcedequantize_ops(
    model_cls, inference_input_type, inference_output_type
):
    model_lce = convert_keras_model(
        model_cls(),
        inference_input_type=inference_input_type,
        inference_output_type=inference_output_type,
    )
    model_lce = strip_lcedequantize_ops(model_lce)
    interpreter = tf.lite.Interpreter(model_content=model_lce)
    output_details = interpreter.get_output_details()
    assert len(output_details) == 1
    assert output_details[0]["dtype"] == tf.int32.as_numpy_dtype


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
