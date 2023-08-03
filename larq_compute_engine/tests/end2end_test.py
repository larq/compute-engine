import functools
import os
import sys
import tempfile
from packaging import version

import larq as lq
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

from larq_compute_engine.mlir.python.converter import (
    convert_keras_model,
    convert_saved_model,
)
from larq_compute_engine.tflite.python.interpreter import Interpreter

from preprocess import preprocess_image_tensor, IMAGE_SIZE


def convert_keras_model_as_saved_model(model, **kwargs):
    with tempfile.TemporaryDirectory() as saved_model_dir:
        model.save(saved_model_dir, save_format="tf")
        return convert_saved_model(saved_model_dir, **kwargs)


def toy_model(binary_quantizer="ste_sign", **kwargs):
    def block(padding, pad_values, activation):
        def dummy(x):
            shortcut = x
            x = lq.layers.QuantConv2D(
                filters=32,
                kernel_size=3,
                padding=padding,
                pad_values=pad_values,
                input_quantizer=binary_quantizer,
                kernel_quantizer=binary_quantizer,
                use_bias=False,
                activation=activation,
            )(x)
            x = tf.keras.layers.BatchNormalization(
                momentum=0.7, beta_initializer="random_normal"
            )(x)
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


def toy_model_sequential(binary_quantizer="ste_sign", **kwargs):
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
                input_quantizer=binary_quantizer,
                kernel_quantizer=binary_quantizer,
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
                input_quantizer=binary_quantizer,
                kernel_quantizer=binary_quantizer,
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
                input_quantizer=binary_quantizer,
                kernel_quantizer=binary_quantizer,
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
    x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
    x = quant(x)
    x = lq.layers.QuantConv2D(
        12, 3, strides=2, input_quantizer="ste_sign", kernel_quantizer="ste_sign"
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
    x = quant(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = quant(x)
    # We do not use a Dense layer or softmax here, because this introduces error,
    # and in this test we want to test the int8 part of our bconvs
    return tf.keras.Model(img, x)


def preprocess(data):
    image = data["image"]
    if len(image.shape) != 3:
        raise ValueError("Input must be of size [height, width, C>0]")
    result = preprocess_image_tensor(tf.convert_to_tensor(image), image_size=IMAGE_SIZE)
    if isinstance(image, np.ndarray):
        return tf.keras.backend.get_value(result)
    return result, data["label"]


def assert_model_output(model_lce, inputs, outputs, rtol, atol):
    interpreter = Interpreter(
        model_lce, num_threads=min(os.cpu_count(), 4), use_reference_bconv=False
    )
    actual_outputs = interpreter.predict(inputs)
    np.testing.assert_allclose(actual_outputs, outputs, rtol=rtol, atol=atol)


@pytest.fixture(scope="module")
def dataset():
    return (
        tfds.load("tf_flowers", split="train", try_gcs=True)
        .map(preprocess, num_parallel_calls=2)
        .take(100)
        .cache()
        .shuffle(100)
        .batch(10)
        .prefetch(1)
    )


def tf_where_binary_quantizer(x):
    return tf.where(x >= 0, tf.ones_like(x), -tf.ones_like(x))


@pytest.mark.parametrize(
    "conversion_function", [convert_keras_model, convert_keras_model_as_saved_model]
)
@pytest.mark.parametrize(
    "model_cls",
    [
        toy_model,
        functools.partial(toy_model, binary_quantizer=tf_where_binary_quantizer),
        toy_model_sequential,
        functools.partial(
            toy_model_sequential, binary_quantizer=tf_where_binary_quantizer
        ),
        toy_model_int8,
    ],
)
def test_simple_model(dataset, conversion_function, model_cls):
    model = model_cls(weights="imagenet")

    # For the untrained models, do a very small amount of training so that the
    # batch norm stats (and, less importantly, the weights) have sensible
    # values.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
    )
    model.fit(dataset, epochs=1)

    model_lce = conversion_function(model)

    if model_cls == toy_model_int8:
        # 0.025 atol is chosen because it allows an off-by-one error in the int8
        # model (with +-3 scales) but not off-by-two.
        rtol = 0.01
        atol = 0.025
    else:
        rtol = 0.001
        atol = 0.001

    # Test on a single batch of images
    inputs = next(iter(dataset.take(1)))[0]
    outputs = model(inputs).numpy()
    assert_model_output(model_lce, inputs.numpy(), outputs, rtol, atol)

    # Test on some random inputs
    input_shape = (10, *model.input.shape[1:])
    inputs = np.random.uniform(-1, 1, size=input_shape).astype(np.float32)
    outputs = model(inputs).numpy()
    assert_model_output(model_lce, inputs, outputs, rtol, atol)


@pytest.mark.parametrize(
    "conversion_function", [convert_keras_model, convert_keras_model_as_saved_model]
)
@pytest.mark.parametrize("model_cls", [toy_model, toy_model_int8])
@pytest.mark.parametrize("inference_input_type", [tf.int8, tf.float32])
@pytest.mark.parametrize("inference_output_type", [tf.int8, tf.float32])
def test_int8_input_output(
    conversion_function, model_cls, inference_input_type, inference_output_type
):
    if version.parse(tf.__version__) < version.parse("2.2"):
        pytest.skip("TensorFlow 2.2 or newer is required for saved model conversion.")

    model_lce = conversion_function(
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
