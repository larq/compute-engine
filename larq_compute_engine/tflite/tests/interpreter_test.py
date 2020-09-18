import sys

import numpy as np
import pytest
import tensorflow as tf

from larq_compute_engine.tflite.python.interpreter import Interpreter


def test_interpreter():
    input_shape = (24, 24, 3)
    x = tf.keras.Input(input_shape)
    model = tf.keras.Model(x, tf.keras.layers.Flatten()(x))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    inputs = np.random.uniform(-1, 1, size=(16, *input_shape)).astype(np.float32)
    expected_outputs = inputs.reshape(16, -1)

    interpreter = Interpreter(converter.convert())
    assert interpreter.input_types == [np.float32]
    assert interpreter.output_types == [np.float32]
    assert interpreter.input_shapes == [(1, *input_shape)]
    assert interpreter.output_shapes == [(1, np.product(input_shape))]

    outputs = interpreter.predict(inputs, 1)
    np.testing.assert_allclose(outputs, expected_outputs)


def test_interpreter_multi_input():
    x = tf.keras.Input((24, 24, 2))
    y = tf.keras.Input((24, 24, 1))
    model = tf.keras.Model(
        [x, y], [tf.keras.layers.Flatten()(x), tf.keras.layers.Flatten()(y)]
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    x_np = np.random.uniform(-1, 1, size=(16, 24, 24, 2)).astype(np.float32)
    y_np = np.random.uniform(-1, 1, size=(16, 24, 24, 1)).astype(np.float32)
    expected_output_x = x_np.reshape(16, -1)
    expected_output_y = y_np.reshape(16, -1)

    interpreter = Interpreter(converter.convert())
    assert interpreter.input_types == [np.float32, np.float32]
    assert interpreter.output_types == [np.float32, np.float32]
    assert interpreter.input_shapes == [(1, 24, 24, 2), (1, 24, 24, 1)]
    assert interpreter.output_shapes == [(1, 24 * 24 * 2), (1, 24 * 24 * 1)]

    output_x, output_y = interpreter.predict([x_np, y_np])
    np.testing.assert_allclose(output_x, expected_output_x)
    np.testing.assert_allclose(output_y, expected_output_y)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
