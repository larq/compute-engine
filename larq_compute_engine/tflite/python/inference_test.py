"""Test for compute engine TF lite Python wrapper"""
import numpy as np
import tensorflow as tf
import larq_compute_engine as lqce

from tflite_runtime.interpreter import Interpreter


# Generate a tf.keras model.
def generate_kerasmodel():
    def shortcut_block(x):
        shortcut = x
        x = tf.keras.layers.Lambda(lqce.bsign)(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        return tf.keras.layers.add([x, shortcut])

    input_shape = (28, 28, 3)
    num_classes = 10
    img_input = tf.keras.layers.Input(shape=input_shape)
    out = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(
        img_input
    )
    out = tf.keras.layers.BatchNormalization(momentum=0.8)(out)
    out = shortcut_block(out)
    out = shortcut_block(out)
    out = shortcut_block(out)
    out = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(out)
    out = tf.keras.layers.GlobalAvgPool2D()(out)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs=img_input, outputs=out)


def generate_data():
    input_shape = (1, 28, 28, 3)
    return np.random.random_sample(input_shape).astype(np.float32)


def test_inference():
    # Get model and random data
    model = generate_kerasmodel()
    data = generate_data()

    # Get result in tensorflow
    tf_result = model.predict(data)

    # Convert to tflite
    keras_file = "/tmp/modelconverter_temporary.h5"
    tf.keras.models.save_model(model, keras_file)
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

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

    # Get tflite result
    tflite_result = interpreter.get_tensor(output_details[0]["index"])

    np.testing.assert_allclose(tflite_result, tf_result, rtol=1e-04)
