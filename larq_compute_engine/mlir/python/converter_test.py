import sys
import unittest
from unittest import mock

import tensorflow as tf
import larq as lq
from tensorflow.python.eager import context

sys.modules["larq_compute_engine.mlir._tf_tfl_flatbuffer"] = mock.MagicMock()
sys.modules[
    "larq_compute_engine.tflite.python.interpreter_wrapper_lite"
] = mock.MagicMock()
sys.modules["larq_compute_engine.mlir.python.tflite_schema"] = mock.MagicMock()

from larq_compute_engine.mlir.python.converter import convert_keras_model
from larq_compute_engine.mlir._tf_tfl_flatbuffer import (
    convert_saved_model_to_tflite_flatbuffer as mocked_saved_model_converter,
)


def get_test_model():
    """Model taken from https://docs.larq.dev/larq/tutorials/mnist/#create-the-model."""

    # All quantized layers except the first will use the same options
    kwargs = {
        "input_quantizer": "ste_sign",
        "kernel_quantizer": "ste_sign",
        "kernel_constraint": "weight_clip",
    }

    model = tf.keras.models.Sequential()

    # In the first layer we only quantize the weights and not the input
    model.add(
        lq.layers.QuantConv2D(
            32,
            (3, 3),
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False,
            input_shape=(28, 28, 1),
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(tf.keras.layers.Flatten())

    model.add(lq.layers.QuantDense(64, use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(tf.keras.layers.Activation("softmax"))
    return model


class TestConverter(unittest.TestCase):
    def test_model(self):
        with context.eager_mode():
            model = get_test_model()
            convert_keras_model(model)
        mocked_saved_model_converter.assert_called_once_with(
            mock.ANY, ["serve"], ["serving_default"], 1, "arm", None
        )

    def test_wrong_arg(self):
        with self.assertRaises(ValueError):
            convert_keras_model("./model.h5")

    def test_target_arg(self):
        with context.eager_mode():
            model = get_test_model()

            # These should work
            convert_keras_model(model, target="arm")
            convert_keras_model(model, target="xcore")

            # Anything else shouldn't
            with self.assertRaises(
                ValueError, msg='Expected `target` to be "arm" or "xcore"'
            ):
                convert_keras_model(model, target="x86")


if __name__ == "__main__":
    unittest.main()
