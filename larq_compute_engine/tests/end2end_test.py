import sys
import pytest
import larq as lq
import larq_zoo as lqz
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from larq_compute_engine.mlir.python.converter import convert_keras_model
from larq_compute_engine.tests._end2end_verify import run_model


def toy_model(**kwargs):
    def block(padding, pad_values, activation):
        def dummy(x):
            shortcut = x
            x = lq.layers.QuantConv2D(
                filters=32,
                kernel_size=3,
                padding=padding,
                pad_values=pad_values,
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


def toy_model_sequential(**kwargs):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((224, 224, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.BatchNormalization(
                gamma_initializer=tf.keras.initializers.RandomNormal(1.0),
                beta_initializer="uniform",
            ),
            # This will be converted to a float->bitpacked binary max pool.
            tf.keras.layers.MaxPooling2D((2, 2)),
            lq.layers.QuantConv2D(
                32,
                (3, 3),
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                padding="same",
                pad_values=1.0,
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(
                # Use an initialiser with mean 0 to test the negative
                # multipliers corner-case.
                gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0),
                beta_initializer="uniform",
            ),
            lq.layers.QuantConv2D(
                32,
                (3, 3),
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                strides=(2, 2),
                padding="same",
                pad_values=1.0,
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(
                # Use an initialiser with mean 0 to test the negative
                # multipliers corner-case.
                gamma_initializer=tf.keras.initializers.RandomNormal(0.0),
                beta_initializer="uniform",
            ),
            # This will be converted to a bitpacked->bitpacked binary max pool.
            # Test some funky filter/stride combination.
            tf.keras.layers.MaxPooling2D((3, 2), strides=(1, 2)),
            lq.layers.QuantConv2D(
                32,
                (3, 3),
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                padding="same",
                pad_values=1.0,
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(
                gamma_initializer=tf.keras.initializers.RandomNormal(1.0),
                beta_initializer="uniform",
            ),
            tf.keras.layers.GlobalAvgPool2D(),
        ]
    )


def quant(x):
    return tf.quantization.fake_quant_with_min_max_vars(x, -3.0, 3.0)


def toy_model_int8(**kwargs):
    img = tf.keras.layers.Input(shape=(224, 224, 3))
    x = quant(img)
    x = lq.layers.QuantConv2D(
        12, 3, input_quantizer="ste_sign", kernel_quantizer="ste_sign", activation=quant
    )(x)
    x = lq.layers.QuantConv2D(
        12, 3, input_quantizer="ste_sign", kernel_quantizer="ste_sign", activation=quant
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = lq.layers.QuantDense(
        10, input_quantizer=quant, kernel_quantizer=quant, activation=quant
    )(x)
    x = tf.keras.layers.Activation("softmax")(x)
    return tf.keras.Model(img, x)


def preprocess(data):
    return lqz.preprocess_input(data["image"])


def assert_model_output(model_lce, inputs, outputs):
    for input, output in zip(inputs, outputs):
        actual_outputs = run_model(model_lce, list(input.flatten()))
        assert len(actual_outputs) > 1
        np.testing.assert_allclose(actual_outputs[0], actual_outputs[1], rtol=1e-5)
        for actual_output in actual_outputs:
            np.testing.assert_allclose(actual_output, output, rtol=0.001, atol=0.25)


@pytest.mark.parametrize(
    "model_cls", [toy_model, toy_model_sequential, toy_model_int8, lqz.sota.QuickNet],
)
def test_simple_model(model_cls):
    model = model_cls(weights="imagenet")
    model_lce = convert_keras_model(
        model, experimental_enable_bitpacked_activations=True
    )

    # Test on the flowers dataset
    dataset = (
        tfds.load("tf_flowers", split="train", try_gcs=True)
        .map(preprocess)
        .shuffle(100)
        .batch(10)
        .take(1)
    )
    inputs = next(tfds.as_numpy(dataset))

    outputs = model(inputs).numpy()
    assert_model_output(model_lce, inputs, outputs)

    # Test on some random inputs
    input_shape = (10, *model.input.shape[1:])
    inputs = np.random.uniform(-1, 1, size=input_shape).astype(np.float32)
    outputs = model(inputs).numpy()
    assert_model_output(model_lce, inputs, outputs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
