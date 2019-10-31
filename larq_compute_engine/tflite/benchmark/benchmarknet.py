import numpy as np
import tensorflow as tf
import larq_compute_engine as lqce

# The HWIO support in TF lite will be dropped at some point.
# But for now it remains usefull for comparison

# Returns both a binary convolution and a normal convolution using the same weights
def BothConv(features, kernel_size, stride):
    def _layerfunc(x):
        in_features = x.shape[3]
        weights = np.random.choice(
            [-1, 1], [kernel_size, kernel_size, in_features, features]
        )
        weights_ohwi = np.moveaxis(weights, 3, 0)
        strides = [1, stride, stride, 1]
        name = f"{in_features}-{features}-{kernel_size}-{stride}"
        out_hwio = lqce.bconv2d(
            x,
            weights,
            strides,
            "SAME",
            data_format="NHWC",
            filter_format="HWIO",
            name=name + "-bin-hwio",
        )
        out_ohwi = lqce.bconv2d(
            x,
            weights_ohwi,
            strides,
            "SAME",
            data_format="NHWC",
            filter_format="OHWI",
            name=name + "-bin-ohwi",
        )
        out_float = tf.nn.conv2d(
            x, weights, strides, "SAME", data_format="NHWC", name=name + "-float"
        )
        return (out_hwio, out_ohwi, out_float)

    return _layerfunc


def benchmark_nets():
    in_shapes = [(56, 56, 64), (28, 28, 128), (14, 14, 256), (7, 7, 512)]
    kernel_sizes = [1, 3, 5]
    strides = [1, 2]

    inputs = [tf.keras.layers.Input(shape=x) for x in in_shapes]
    outputs_hwio = []
    outputs_ohwi = []
    outputs_float = []
    for img in inputs:
        f = img.shape[3]
        for s in strides:
            for k in kernel_sizes:
                out_hwio, out_ohwi, out_float = BothConv(
                    features=f, kernel_size=k, stride=s
                )(img)
                outputs_hwio.append(out_hwio)
                outputs_ohwi.append(out_ohwi)
                outputs_float.append(out_float)

    # We have to add some dummy op at the end because the final op in each output
    # is renamed to "Identity_k" for some integer k, and that makes it
    # hard to read benchmark results.
    # Therefore we add a Flatten op
    outputs_hwio = [tf.reshape(x, [-1]) for x in outputs_hwio]
    outputs_ohwi = [tf.reshape(x, [-1]) for x in outputs_ohwi]
    outputs_float = [tf.reshape(x, [-1]) for x in outputs_float]

    model_hwio = tf.keras.Model(inputs=inputs, outputs=outputs_hwio)
    model_ohwi = tf.keras.Model(inputs=inputs, outputs=outputs_ohwi)
    model_float = tf.keras.Model(inputs=inputs, outputs=outputs_float)

    return model_hwio, model_ohwi, model_float


model_hwio, model_ohwi, model_float = benchmark_nets()

conv = lqce.ModelConverter(model_hwio)
conv.convert("benchmarknet_hwio.tflite")

conv = lqce.ModelConverter(model_ohwi)
conv.convert("benchmarknet_ohwi.tflite")

conv = lqce.ModelConverter(model_float)
conv.convert("benchmarknet_float.tflite")
