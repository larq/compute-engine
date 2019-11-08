import numpy as np
import tensorflow as tf
import larq_compute_engine as lqce
import itertools
import os

# The HWIO support in TF lite will be dropped at some point.
# But for now it remains usefull for comparison

# Returns OHWI binary convolution and a float convolution using the same weights
def AllConv(features, kernel_size, stride):
    def _layerfunc(x):
        in_features = x.shape[3]
        weights = np.random.choice(
            [-1, 1], [kernel_size, kernel_size, in_features, features]
        )
        weights_ohwi = np.moveaxis(weights, 3, 0)
        strides = [1, stride, stride, 1]
        name = f"{in_features}-{features}-{kernel_size}-{stride}"
        out_ohwi = lqce.bconv2d(
            x,
            weights_ohwi,
            strides,
            "SAME",
            data_format="NHWC",
            filter_format="OHWI",
            name=name + "-bin",
        )
        out_float = tf.nn.conv2d(
            x, weights, strides, "SAME", data_format="NHWC", name=name + "-float"
        )
        return (out_ohwi, out_float)

    return _layerfunc


def generate_benchmark_nets():
    in_shapes = [(56, 56, 64), (28, 28, 128), (14, 14, 256), (7, 7, 512)]
    kernel_sizes = [1, 3, 5]
    strides = [1, 2]

    args_lists = [in_shapes, kernel_sizes, strides]

    if not os.path.exists("benchmarking_models"):
        os.makedirs("benchmarking_models")

    for in_shape, kernel_size, stride in itertools.product(*args_lists):
        img = tf.keras.layers.Input(shape=in_shape)
        # Same number of output features as input features
        f = in_shape[2]

        outs = AllConv(features=f, kernel_size=kernel_size, stride=stride)(img)

        for out, label in zip(outs, ["bin", "float"]):
            model = tf.keras.Model(inputs=img, outputs=out)

            filename = f"benchmarking_models/benchmarknet_{label}_input_{in_shape[0]}_kernel_{kernel_size}_stride_{stride}_features_{f}.tflite"

            conv = lqce.ModelConverter(model)
            conv.convert(filename)


generate_benchmark_nets()
