"""Tests for TF lite model conversion."""
import tensorflow as tf
import larq_compute_engine as lqce

# Generate a tf.keras model.
def testnet(use_batchnorm=False, number_of_bsign=1):
    input_shape = (28, 28, 3)
    num_classes = 10
    img_input = tf.keras.layers.Input(shape=input_shape)
    out = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(
        img_input
    )
    if number_of_bsign >= 1:
        out = tf.keras.layers.Lambda(lqce.bsign)(out)
    if use_batchnorm:
        out = tf.keras.layers.BatchNormalization(momentum=0.8)(out)
    out = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(out)
    if number_of_bsign >= 2:
        out = tf.keras.layers.Lambda(lqce.bsign)(out)
    out = tf.keras.layers.GlobalAvgPool2D()(out)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs=img_input, outputs=out)


model = testnet(use_batchnorm=True, number_of_bsign=2)

conv = lqce.ModelConverter(model)
assert conv.convert("/tmp/testnet_converted_model.tflite")
