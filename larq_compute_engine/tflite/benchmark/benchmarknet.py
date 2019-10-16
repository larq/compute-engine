import numpy as np
import tensorflow as tf
import larq_compute_engine as lqce


# Applies both a binary convolution and a normal convolution, in parallel
def BothConv(features, kernel_size, stride):
    def _layerfunc(x):
        in_features = x.shape[3]
        weights = np.random.choice(
            [-1, 1], [kernel_size, kernel_size, in_features, features]
        )
        strides = [1, stride, stride, 1]
        name = f"{features}-{kernel_size}-{stride}"
        out1 = lqce.bconv2d(
            x, weights, strides, "SAME", data_format="NHWC", name=name + "-binary"
        )
        out2 = tf.nn.conv2d(
            x, weights, strides, "SAME", data_format="NHWC", name=name + "-float"
        )
        return (out1, out2)

    return _layerfunc


def benchmark_net():
    in_shapes = [(56, 56, 64), (28, 28, 128), (14, 14, 256), (7, 7, 512)]
    kernel_sizes = [1, 3, 5]
    strides = [1, 2]

    inputs = [tf.keras.layers.Input(shape=x) for x in in_shapes]
    outputs = []
    for img in inputs:
        f = img.shape[3]
        for s in strides:
            for k in kernel_sizes:
                x1, x2 = BothConv(features=f, kernel_size=k, stride=s)(img)
                outputs.extend([x1, x2])

    # We have to add some dummy op at the end because the final op in each output
    # is renamed to "Identity_k" for some integer k, and that makes it
    # hard to read benchmark results.
    # Therefore we add a Flatten op
    outputs = [tf.reshape(x, [-1]) for x in outputs]

    return tf.keras.Model(inputs=inputs, outputs=outputs)


model = benchmark_net()
conv = lqce.ModelConverter(model)
conv.convert("benchmarknet.tflite")
