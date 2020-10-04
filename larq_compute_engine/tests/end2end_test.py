import math
import os
import sys

import larq as lq
import larq_zoo as lqz
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

from larq_compute_engine.mlir.python.converter import convert_keras_model
from larq_compute_engine.tflite.python.interpreter import Interpreter


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
            x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
            return tf.keras.layers.add([x, shortcut])

        return dummy

    img_input = tf.keras.layers.Input(shape=(224, 224, 3))
    out = img_input
    out = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(
        out
    )

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
            tf.keras.layers.BatchNormalization(momentum=0.7),
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
                momentum=0.7,
                # Use a Gamma initialiser with mean -0.1 to test the negative
                # multipliers corner-case.
                gamma_initializer=tf.keras.initializers.RandomNormal(mean=-0.1),
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
                momentum=0.7,
                # Use a Gamma initialiser with mean -0.1 to test the negative
                # multipliers corner-case.
                gamma_initializer=tf.keras.initializers.RandomNormal(mean=-0.1),
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
            tf.keras.layers.BatchNormalization(momentum=0.7),
            tf.keras.layers.GlobalAvgPool2D(),
        ]
    )


def quant(x):
    return tf.quantization.fake_quant_with_min_max_vars(x, -3.0, 3.0)


def toy_model_int8(**kwargs):
    img = tf.keras.layers.Input(shape=(224, 224, 3))
    x = quant(img)
    x = lq.layers.QuantConv2D(
        12, 3, strides=2, input_quantizer="ste_sign", kernel_quantizer="ste_sign"
    )(x)
    # Make sure the typical output is in the (-3, 3) range
    # by dividing by sqrt(filter_height * filter_width * input_channels)
    x = tf.keras.layers.BatchNormalization(
        gamma_initializer=tf.keras.initializers.RandomNormal(
            1.0 / math.sqrt(3 * 3 * 3), stddev=0.1 / math.sqrt(3 * 3 * 3)
        ),
        beta_initializer="uniform",
    )(x)
    x = quant(x)
    x = lq.layers.QuantConv2D(
        12, 3, strides=2, input_quantizer="ste_sign", kernel_quantizer="ste_sign"
    )(x)
    # Make sure the typical output is in the (-3, 3) range
    # by dividing by sqrt(filter_height * filter_width * input_channels)
    x = tf.keras.layers.BatchNormalization(
        gamma_initializer=tf.keras.initializers.RandomNormal(
            1.0 / math.sqrt(3 * 3 * 12), stddev=0.1 / math.sqrt(3 * 3 * 12)
        ),
        beta_initializer="uniform",
    )(x)
    x = quant(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = quant(x)
    # We do not use a Dense layer or softmax here, because this introduces error,
    # and in this test we want to test the int8 part of our bconvs
    return tf.keras.Model(img, x)


def preprocess(data):
    return lqz.preprocess_input(data["image"]), data["label"]


def assert_model_output(model_lce, inputs, outputs):
    interpreter = Interpreter(model_lce, num_threads=min(os.cpu_count(), 4))
    actual_outputs = interpreter.predict(inputs)
    np.testing.assert_allclose(actual_outputs, outputs, rtol=0.05, atol=0.125)


@pytest.mark.parametrize(
    "model_cls",
    [toy_model, toy_model_sequential, toy_model_int8, lqz.sota.QuickNet],
)
def test_simple_model(model_cls):
    # Test on the Oxford flowers dataset
    dataset = (
        tfds.load("tf_flowers", split="train", try_gcs=True)
        .map(preprocess)
        .shuffle(100)
        .batch(10)
    )

    model = model_cls(weights="imagenet")

    # For the untrained models, do a very small amount of training so that the
    # batch norm stats (and, less importantly, the weights) have sensible
    # values.
    if model_cls != lqz.sota.QuickNet:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
        )
        model.fit(dataset, epochs=1, steps_per_epoch=10)

    model_lce = convert_keras_model(
        model, experimental_enable_bitpacked_activations=True
    )

    # Test on a single batch of images
    inputs = next(tfds.as_numpy(dataset.map(lambda *data: data[0]).take(1)))
    outputs = model(inputs).numpy()
    assert_model_output(model_lce, inputs, outputs)

    # Test on some random inputs
    input_shape = (10, *model.input.shape[1:])
    inputs = np.random.uniform(-1, 1, size=input_shape).astype(np.float32)
    outputs = model(inputs).numpy()
    assert_model_output(model_lce, inputs, outputs)


@pytest.mark.parametrize("model_cls", [toy_model, toy_model_int8])
@pytest.mark.parametrize("inference_input_type", [tf.int8, tf.float32])
@pytest.mark.parametrize("inference_output_type", [tf.int8, tf.float32])
def test_int8_input_output(model_cls, inference_input_type, inference_output_type):
    model_lce = convert_keras_model(
        model_cls(),
        inference_input_type=inference_input_type,
        inference_output_type=inference_output_type,
        experimental_default_int8_range=(-6.0, 6.0) if model_cls == toy_model else None,
    )
    interpreter = tf.lite.Interpreter(model_content=model_lce)
    input_details = interpreter.get_input_details()
    assert len(input_details) == 1
    assert input_details[0]["dtype"] == inference_input_type.as_numpy_dtype
    output_details = interpreter.get_output_details()
    assert len(output_details) == 1
    assert output_details[0]["dtype"] == inference_output_type.as_numpy_dtype


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
