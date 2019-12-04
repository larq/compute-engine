import tensorflow as tf
import larq as lq
from inference_test_base import InferenceTest


class LargeModelTest(InferenceTest):
    def test_largemodel(self):
        def shortcut_block(x):
            shortcut = x
            # Larq quantized convolution layer
            x = lq.layers.QuantConv2D(
                64,
                3,
                padding="same",
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
                use_bias=False,
            )(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
            return tf.keras.layers.add([x, shortcut])

        img_input = tf.keras.layers.Input(shape=(28, 28, 3))
        out = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(
            img_input
        )
        out = tf.keras.layers.BatchNormalization(momentum=0.8)(out)
        out = shortcut_block(out)
        out = shortcut_block(out)
        out = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(out)
        out = tf.keras.layers.GlobalAvgPool2D()(out)
        out = tf.keras.layers.Dense(10, activation="softmax")(out)

        # model = tf.keras.Model(inputs=img_input, outputs=out)
        # self.run_inference(model)


if __name__ == "__main__":
    tf.test.main()
