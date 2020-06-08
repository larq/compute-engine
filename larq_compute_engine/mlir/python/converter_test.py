import sys
import unittest
from unittest import mock

import larq_zoo as lqz
from tensorflow.python.eager import context

sys.modules["larq_compute_engine.mlir._graphdef_tfl_flatbuffer"] = mock.MagicMock()

from larq_compute_engine.mlir.python.converter import convert_keras_model
from larq_compute_engine.mlir._graphdef_tfl_flatbuffer import (
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
            None,
            False,
        )


if __name__ == "__main__":
    unittest.main()
