from larq_compute_engine.tflite.python.interpreter_base import InterpreterBase

__all__ = ["Interpreter"]


class Interpreter(InterpreterBase):
    """Interpreter interface for Larq Compute Engine Models.

    !!! warning
        The Larq Compute Engine is optimized for ARM v8, which is used by devices
        such as a Raspberry Pi or Android phones. Running this interpreter on any
        other architecture (e.g. x86) will fall back on the reference kernels, meaning
        it will return correct outputs, but is not optimized for speed in any way!

    !!! example
        ```python
        lce_model = convert_keras_model(model)
        interpreter = Interpreter(lce_model)
        interpreter.predict(input_data, verbose=1)
        ```

    See the base class `InterpreterBase` for the full interface.

    # Arguments
        flatbuffer_model: A serialized Larq Compute Engine model in the flatbuffer format.
        num_threads: The number of threads used by the interpreter.
        use_reference_bconv: When True, uses the reference implementation of LceBconv2d.
    """

    def __init__(
        self,
        flatbuffer_model: bytes,
        num_threads: int = 1,
        use_reference_bconv: bool = False,
    ):
        from larq_compute_engine.tflite.python import interpreter_wrapper_lite

        super().__init__(
            interpreter_wrapper_lite.LiteInterpreter(
                flatbuffer_model, num_threads, use_reference_bconv
            )
        )
