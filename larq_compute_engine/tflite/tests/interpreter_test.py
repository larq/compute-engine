import sys

import numpy as np
import pytest
import tensorflow as tf

from larq_compute_engine.tflite.python.interpreter import Interpreter


@pytest.mark.parametrize("input_shapes", [[(24,)], [(24, 24, 3)], [(16, 16, 1), (3,)]])
def test_interpreter(input_shapes):
    input_tensors = [tf.keras.Input(shape) for shape in input_shapes]
    output_tensors = [(i + 1) * x for i, x in enumerate(input_tensors)]
    model = tf.keras.Model(input_tensors, output_tensors)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    flatbuffer_model = converter.convert()

    shapes = [(1, *shape) for shape in input_shapes]
    types = [np.float32] * len(input_shapes)
    batch_size = 2
    inputs = [
        [np.random.uniform(-1, 1, size=shape).astype(np.float32) for shape in shapes]
        for _ in range(batch_size)
    ]
    expected_outputs = [[(i + 1) * x for i, x in enumerate(inp)] for inp in inputs]
    if len(shapes) == 1:
        inputs = [inp[0] for inp in inputs]
        expected_outputs = [out[0] for out in expected_outputs]

    interpreter = Interpreter(flatbuffer_model)
    assert interpreter.input_types == types
    assert interpreter.output_types == types
    assert interpreter.input_shapes == shapes
    assert interpreter.output_shapes == shapes

    outputs = interpreter.predict(inputs)
    assert len(outputs) == len(expected_outputs)
    for output, expected_output in zip(outputs, expected_outputs):
        if isinstance(expected_output, list):
            for out, expected_out in zip(output, expected_output):
                np.testing.assert_allclose(out, expected_out, rtol=0.001, atol=0.25)
        else:
            np.testing.assert_allclose(output, expected_output, rtol=0.001, atol=0.25)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
