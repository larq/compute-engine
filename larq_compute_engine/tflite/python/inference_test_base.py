"""Test for compute engine TF lite Python wrapper"""
import numpy as np
import tensorflow as tf
import larq_compute_engine as lqce

from tflite_runtime.interpreter import Interpreter

# Base class for testing a model and comparing tf with tflite
class InferenceTest(tf.test.TestCase):
    def run_inference(self, model):
        # Set batch dimension to 1
        input_shape = (1,) + model.input_shape[1:]
        # Generate data
        data = np.random.random_sample(input_shape).astype(np.float32)

        # Get result in tensorflow
        tf_result = model.predict(data)

        # Convert to tflite
        converter = lqce.ModelConverter(model)
        tflite_model = converter.convert()

        # Setup tflite
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        tflite_input_type = input_details[0]["dtype"]
        tflite_input_shape = input_details[0]["shape"]

        self.assertAllEqual(tflite_input_shape, data.shape)
        self.assertEqual(tflite_input_type, data.dtype)

        interpreter.set_tensor(input_details[0]["index"], data)

        # Run tflite
        interpreter.invoke()

        # Get tflite result
        tflite_result = interpreter.get_tensor(output_details[0]["index"])

        self.assertAllClose(tf_result, tflite_result, rtol=1e-04, atol=1e-05)
