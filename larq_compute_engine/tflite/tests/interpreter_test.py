import sys

import numpy as np
import pytest
import tensorflow as tf

from larq_compute_engine.tflite.python.interpreter import Interpreter


@pytest.mark.parametrize("use_iterator", [True, False])
def test_interpreter(use_iterator):
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

    def input_fn():
        if use_iterator:
            return (input for input in inputs)
        return inputs

    outputs = interpreter.predict(input_fn(), 1)
    np.testing.assert_allclose(outputs, expected_outputs)


@pytest.mark.parametrize("use_iterator", [True, False])
def test_interpreter_multi_input(use_iterator):
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

    interpreter = Interpreter(converter.convert(), num_threads=2)
    assert interpreter.input_types == [np.float32, np.float32]
    assert interpreter.output_types == [np.float32, np.float32]
    assert interpreter.input_shapes == [(1, 24, 24, 1), (1, 24, 24, 2)]
    assert sorted(interpreter.output_shapes) == [(1, 24 * 24 * 1), (1, 24 * 24 * 2)]

    def input_fn():
        if use_iterator:
            return ([y, x] for x, y in zip(x_np, y_np))
        return [y_np, x_np]

    output_x, output_y = interpreter.predict(input_fn())
    # Output order is not deterministic, decide based on shape
    if output_y.shape == expected_output_x.shape:
        output_x, output_y = output_y, output_x
    np.testing.assert_allclose(output_x, expected_output_x)
    np.testing.assert_allclose(output_y, expected_output_y)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
