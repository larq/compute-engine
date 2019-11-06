"""Test for compute engine TF lite Python wrapper"""
import numpy as np
import tensorflow as tf
import larq_compute_engine as lqce
import itertools

from tflite_runtime.interpreter import Interpreter


def _create_bconv2d_layer(
    bconv_op, in_channels, filters, kernel_size, hw_strides=(1, 1), padding="VALID"
):
    strides = [1] + list(hw_strides) + [1]
    fshape = [kernel_size, kernel_size, in_channels, filters]
    filt_tf = np.random.choice([-1, 1], fshape)
    filt_lite = np.copy(np.moveaxis(filt_tf, 3, 0))

    def _layer_tf(x):
        return bconv_op(
            x, filt_tf, strides, padding, data_format="NHWC", filter_format="HWIO"
        )

    def _layer_lite(x):
        return bconv_op(
            x, filt_lite, strides, padding, data_format="NHWC", filter_format="OHWI"
        )

    return _layer_tf, _layer_lite


def _create_sample_bconv_model(
    bconv_op, input_shape, filters, kernel_size, strides, padding
):
    layer_tf, layer_lite = _create_bconv2d_layer(
        bconv_op, input_shape[2], filters, kernel_size, strides, padding
    )

    img_input = tf.keras.layers.Input(shape=input_shape)

    out_tf = layer_tf(img_input)
    out_lite = layer_lite(img_input)

    model_tf = tf.keras.Model(inputs=img_input, outputs=out_tf)
    model_lite = tf.keras.Model(inputs=img_input, outputs=out_lite)

    return model_tf, model_lite


def invoke_inference(model, data):
    # Convert to tflite
    converter = lqce.ModelConverter(model)
    tflite_model = converter.convert()

    # Setup tflite
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_type = input_details[0]["dtype"]
    input_shape = input_details[0]["shape"]

    np.testing.assert_equal(input_shape, data.shape)
    np.testing.assert_equal(input_type, data.dtype)

    interpreter.set_tensor(input_details[0]["index"], data)

    # Run tflite
    interpreter.invoke()

    # return tflite result
    return interpreter.get_tensor(output_details[0]["index"])


class BConv2DTest(tf.test.TestCase):
    def __test_bconv_op_inference(self, bconv_op):
        data_types = [np.float32]
        in_channels = [1, 3]
        out_channels = [1, 4]
        input_sizes = [14]
        kernel_sizes = [3, 5]
        hw_strides = [[1, 1], [2, 2]]
        paddings = ["VALID", "SAME"]

        args_lists = [
            data_types,
            in_channels,
            out_channels,
            input_sizes,
            kernel_sizes,
            hw_strides,
            paddings,
        ]
        for args in itertools.product(*args_lists):
            print(args)
            data_type, in_channel, out_channel, input_size, kernel_size, strides, padding = (
                args
            )
            input_shape = [input_size, input_size, in_channel]
            model_tf, model_lite = _create_sample_bconv_model(
                bconv_op, input_shape, out_channel, kernel_size, strides, padding
            )

            input_shape = [1] + input_shape
            input_data = np.random.choice([-1, 1], input_shape).astype(data_type)

            tflite_result = invoke_inference(model_lite, input_data)
            tf_result = model_tf.predict(input_data)

            self.assertAllClose(tflite_result, tf_result)

    def test_bconv2d8(self):
        self.__test_bconv_op_inference(lqce.bconv2d8)

    def test_bconv2d32(self):
        self.__test_bconv_op_inference(lqce.bconv2d32)

    def test_bconv2d64(self):
        self.__test_bconv_op_inference(lqce.bconv2d64)


if __name__ == "__main__":
    tf.test.main()
