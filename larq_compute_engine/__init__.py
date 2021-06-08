from larq_compute_engine.mlir.python.converter import (
    convert_keras_model,
    convert_saved_model,
)
from larq_compute_engine.tflite.python import interpreter as testing

try:
    from importlib import metadata  # type: ignore
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata  # type: ignore

__version__ = metadata.version("larq_compute_engine")

__all__ = ["convert_keras_model", "convert_saved_model", "testing"]
