import sys
import pytest
import larq as lq
import larq_zoo as lqz
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from larq_compute_engine.mlir.python.converter import convert_keras_model
from larq_compute_engine.mlir._end2end_verify import run_model


def toy_model(**kwargs):
    def block(padding, pad_values, activation):
        def dummy(x):
            shortcut = x
            x = lq.layers.QuantConv2D(
                filters=32,
                kernel_size=3,
                padding=padding,
                # pad_values=pad_values,
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                use_bias=False,
                activation=activation,
            )(x)
            x = tf.keras.layers.BatchNormalization(
                gamma_initializer=tf.keras.initializers.RandomNormal(1.0),
                beta_initializer="uniform",
            )(x)
            return tf.keras.layers.add([x, shortcut])

        return dummy

    img_input = tf.keras.layers.Input(shape=(224, 224, 3))
    out = img_input
    out = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(out)

    # Test zero-padding
    out = block("same", 0.0, "relu")(out)
    # Test one-padding
    out = block("same", 1.0, "relu")(out)
    # Test no activation function
    out = block("same", 1.0, None)(out)

    out = tf.keras.layers.GlobalAvgPool2D()(out)
    return tf.keras.Model(inputs=img_input, outputs=out)


def preprocess(data):
    return lqz.preprocess_input(data["image"])


@pytest.mark.parametrize(
    "model_cls", [toy_model, lqz.BinaryResNetE18,],
)
def test_simple_model(model_cls):
    model = model_cls(weights="imagenet")
    model_lce = convert_keras_model(model)

    # Test on the flowers dataset
    dataset = (
        tfds.load("oxford_flowers102", split=tfds.Split.VALIDATION)
        .map(preprocess)
        .shuffle(512)
        .take(10)
        .batch(1)
    )
    for inputs in tfds.as_numpy(dataset):
        outputs = model(inputs).numpy().flatten()
        actual_outs = run_model(model_lce, list(inputs.flatten()))
        for actual_out in actual_outs:
            np.testing.assert_allclose(actual_out, outputs, rtol=0.001, atol=0.2)

    # Test on some random inputs
    for _ in range(10):
        inputshape = (1,) + tuple(model.input.shape)[1:]
        inputs = 2.0 * np.random.random(inputshape) - 1.0
        outputs = model(inputs).numpy().flatten()
        actual_outs = run_model(model_lce, list(inputs.flatten()))
        for actual_out in actual_outs:
            np.testing.assert_allclose(actual_out, outputs, rtol=0.001, atol=0.2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
