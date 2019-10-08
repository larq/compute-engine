"""Examples of TF lite model conversion."""
import tensorflow as tf
import larq_compute_engine as lqce
import larq_zoo as lqz

# Generate a tf.keras model with our custom bsign op
def testnet():
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


model = testnet()

conv = lqce.ModelConverter(model)
conv.convert("testnet_converted_model.tflite")


# Example of converting some models from Larq Zoo to TF lite

zoo_models = [
    ("binaryalexnet", lqz.BinaryAlexNet),
    ("birealnet", lqz.BiRealNet),
    ("xnornet", lqz.XNORNet),
    ("binarydensenet45", lqz.BinaryDenseNet45),
    ("dorefanet", lqz.DoReFaNet),
    ("binaryresnete18", lqz.BinaryResNetE18),
]

for (name, modelfunc) in zoo_models:
    tf.keras.backend.clear_session()
    print(f"Converting {name}")
    model = modelfunc(weights="imagenet")
    conv = lqce.ModelConverter(model)
    conv.convert(f"{name}.tflite")
