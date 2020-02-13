import sys
import pytest
import larq_zoo as lqz
from tensorflow.python.eager import context

from larq_compute_engine.mlir.python.converter import convert_keras_model


@pytest.fixture
def eager_mode():
    """pytest fixture for running test in eager mode"""
    with context.eager_mode():
        yield


@pytest.mark.parametrize(
    "model_cls", [lqz.BinaryResNetE18, lqz.BinaryDenseNet28,],
)
def test_larq_zoo_models(eager_mode, model_cls):
    model = model_cls(weights=None)
    convert_keras_model(model)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
