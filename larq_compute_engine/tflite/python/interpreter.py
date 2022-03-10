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

    # Arguments
        flatbuffer_model: A serialized Larq Compute Engine model in the flatbuffer format.
        num_threads: The number of threads used by the interpreter.
        use_reference_bconv: When True, uses the reference implementation of LceBconv2d.
        use_indirect_bgemm: When True, uses the optimized indirect BGEMM kernel of LceBconv2d.
        use_xnnpack: When True, uses the XNNPack delegate of TFLite.

    # Attributes
        input_types: Returns a list of input types.
        input_shapes: Returns a list of input shapes.
        input_scales: Returns a list of input scales.
        input_zero_points: Returns a list of input zero points.
        output_types: Returns a list of output types.
        output_shapes: Returns a list of output shapes.
        output_scales: Returns a list of input scales.
        output_zero_points: Returns a list of input zero points.
    """

    def __init__(
        self,
        flatbuffer_model: bytes,
        num_threads: int = 1,
        use_reference_bconv: bool = False,
        use_indirect_bgemm: bool = False,
        use_xnnpack: bool = False,
    ):
        from larq_compute_engine.tflite.python import interpreter_wrapper_lite

        super().__init__(
            interpreter_wrapper_lite.LiteInterpreter(
                flatbuffer_model,
                num_threads,
                use_reference_bconv,
                use_indirect_bgemm,
                use_xnnpack,
            )
        )
