"""Test for compute engine TF lite Python wrapper"""
import numpy as np
import os
import sys
import pytest

from tflite_runtime.interpreter import Interpreter


def get_filenames():
    (_, _, filenames) = next(os.walk("testing_models/"))
    return [f"testing_models/{name}" for name in filenames]


@pytest.mark.parametrize("filename", get_filenames())
def test_bconv2d_op_inference(filename):
    # Setup tflite
    interpreter = Interpreter(model_path=filename)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_type = input_details[0]["dtype"]
    input_shape = input_details[0]["shape"]

    # Generate random data
    input_data = np.random.choice([-1, 1], input_shape).astype(input_type)

    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Run tflite
    interpreter.invoke()

    # Get results
    outputs = [interpreter.get_tensor(detail["index"]) for detail in output_details]

    # Check if they are equal
    np.testing.assert_allclose(outputs[0], outputs[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
