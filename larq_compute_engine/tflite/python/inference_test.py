"""Test for compute engine TF lite Python wrapper

Based on tensorflow/tensorflow/lite/examples/python/label_image.py
"""
import numpy as np

from tflite_runtime.interpreter import Interpreter


def test_inference():
    # Currently this has to be run from the root directory of the repository
    interpreter = Interpreter("examples/example_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_type = input_details[0]["dtype"]
    input_shape = input_details[0]["shape"]

    # Generate input_data: tensor filled as [0, 1, 2, 3,... ]
    # and then reshaped to input shape.
    input_total_elements = np.product(input_shape)
    input_data = np.arange(0, input_total_elements, dtype=input_type)
    input_data = input_data.reshape(input_shape)

    interpreter.set_tensor(input_details[0]["index"], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    results = np.squeeze(output_data)

    np.testing.assert_allclose(results, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
