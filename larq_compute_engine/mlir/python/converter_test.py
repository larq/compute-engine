import sys
import unittest
from unittest import mock

import larq_zoo as lqz
from tensorflow.python.eager import context

sys.modules["importlib.metadata"] = mock.MagicMock()
sys.modules["importlib_metadata"] = mock.MagicMock()
sys.modules["larq_compute_engine.mlir._tf_tfl_flatbuffer"] = mock.MagicMock()
sys.modules[
    "larq_compute_engine.tflite.python.interpreter_wrapper_lite"
] = mock.MagicMock()
sys.modules["larq_compute_engine.mlir.python.tflite_schema"] = mock.MagicMock()

from larq_compute_engine.mlir.python.converter import convert_keras_model
from larq_compute_engine.mlir._tf_tfl_flatbuffer import (
    convert_graphdef_to_tflite_flatbuffer as mocked_converter,
)


class TestConverter(unittest.TestCase):
    def test_larq_zoo_models(self):
        with context.eager_mode():
            model = lqz.sota.QuickNet(weights=None)
            convert_keras_model(model)
        mocked_converter.assert_called_once_with(
            mock.ANY,
            ["input_1"],
            ["DT_FLOAT"],
            [[1, 224, 224, 3]],
            ["Identity"],
            False,
            "arm",
            None,
            False,
        )

    def test_wrong_arg(self):
        with self.assertRaises(ValueError):
            convert_keras_model("./model.h5")

    def test_target_arg(self):
        with context.eager_mode():
            model = lqz.sota.QuickNet(weights=None)

            # These should work
            convert_keras_model(model, target="arm")
            convert_keras_model(model, target="xcore")

            # Anything else shouldn't
            with self.assertRaises(
                ValueError, msg='Expected `target` to be "arm" or "xcore"'
            ):
                convert_keras_model(model, target="x86")


if __name__ == "__main__":
    unittest.main()
