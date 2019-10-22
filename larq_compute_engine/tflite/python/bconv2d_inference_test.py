"""Test for compute engine TF lite Python wrapper"""
import numpy as np
import tensorflow as tf
import larq_compute_engine as lqce
import itertools

from tflite_runtime.interpreter import Interpreter


def _create_bconv2d_layer(
    bconv_op,
    x,
    filters,
    kernel_size,
    hw_strides=(1, 1),
    padding="VALID",
    data_format="NHWC",
):
    if data_format == "NHWC":
        in_channel = x.shape[3]
        strides = [1] + list(hw_strides) + [1]
    else:
        raise ValueError("Unknown data_format: " + str(data_format))
    sample_list = [-1, 1]
    fshape = [kernel_size, kernel_size, in_channel, filters]
    filt = np.random.choice(sample_list, np.prod(fshape))
    filt = np.reshape(filt, fshape)
    bconvop = lqce.bconv2d(x, filt, strides, padding, data_format=data_format)
    return bconvop


def _create_sample_bconv_model(
    bconv_op, input_shape, filters, kernel_size, strides, padding, data_format
):
    img_input = tf.keras.layers.Input(shape=input_shape)
    out = _create_bconv2d_layer(
        bconv_op, img_input, filters, kernel_size, strides, padding, data_format
    )
    return tf.keras.Model(inputs=img_input, outputs=out)


def invoke_inference(model, data):
    # Convert to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.allow_custom_ops = True
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
        data_formats = ["NHWC"]
        in_channels = [1, 3]
        out_channels = [1, 4]
        input_sizes = [28]
        kernel_sizes = [3, 5]
        hw_strides = [[1, 1], [2, 2]]
        paddings = ["VALID", "SAME"]

        args_lists = [
            data_types,
            data_formats,
            in_channels,
            out_channels,
            input_sizes,
            kernel_sizes,
            hw_strides,
            paddings,
        ]
        for args in itertools.product(*args_lists):
            print(args)
            data_type, data_format, in_channel, out_channel, input_size, kernel_size, strides, padding = (
                args
            )
            input_shape = [input_size, input_size, in_channel]
            model = _create_sample_bconv_model(
                bconv_op,
                input_shape,
                out_channel,
                kernel_size,
                strides,
                padding,
                data_format,
            )

            input_shape = [1] + input_shape
            input_data = np.random.random_sample(input_shape).astype(data_type)

            tflite_result = invoke_inference(model, input_data)
            tf_result = model.predict(input_data)

            self.assertAllClose(tflite_result, tf_result)

    def test_bconv2d8(self):
        self.__test_bconv_op_inference(lqce.bconv2d8)

    def test_bconv2d32(self):
        self.__test_bconv_op_inference(lqce.bconv2d32)

    def test_bconv2d64(self):
        self.__test_bconv_op_inference(lqce.bconv2d64)


if __name__ == "__main__":
    tf.test.main()
