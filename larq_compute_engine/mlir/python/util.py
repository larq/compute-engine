import copy

import flatbuffers
import tensorflow as tf

from larq_compute_engine.mlir.python import tflite_schema


_MAP_TFLITE_ENUM_TO_TF_TYPES = {
    0: tf.float32,
    1: tf.float16,
    2: tf.int32,
    3: tf.uint8,
    4: tf.int64,
    5: tf.string,
    6: tf.bool,
    7: tf.int16,
    8: tf.complex64,
    9: tf.int8,
    10: tf.float64,
    11: tf.complex128,
}

_TFLITE_FILE_IDENTIFIER = b"TFL3"


def _convert_tflite_enum_type_to_tf_type(tflite_enum_type):
    """Converts tflite enum type (eg: 0) to tf type (eg: tf.float32)."""
    tf_type = _MAP_TFLITE_ENUM_TO_TF_TYPES.get(tflite_enum_type)
    if tf_type is None:
        raise ValueError(
            "Unsupported enum {}. The valid map of enum to tf.dtype is : {}".format(
                tflite_enum_type, _MAP_TFLITE_ENUM_TO_TF_TYPES
            )
        )
    return tf_type


def _convert_model_from_bytearray_to_object(model_bytearray):
    """Converts a tflite model from a bytearray into a parsable object."""
    model_object = tflite_schema.Model.GetRootAsModel(model_bytearray, 0)
    model_object = tflite_schema.ModelT.InitFromObj(model_object)
    model_object = copy.deepcopy(model_object)
    model_object.subgraphs[0].inputs[0] = model_object.subgraphs[0].inputs[0]
    return model_object


def _convert_model_from_object_to_bytearray(model_object):
    """Converts a tflite model from a parsable object into a bytearray."""
    # Initial size of the buffer, which will grow automatically if needed
    builder = flatbuffers.Builder(1024)
    model_offset = model_object.Pack(builder)
    builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
    return bytes(builder.Output())


def _update_signature_def_tensors(tensor_maps, map_old_to_new_tensors):
    """Update the tensors in the SignatureDef's TensorMaps."""
    for i in range(len(tensor_maps)):
        if tensor_maps[i].tensorIndex in map_old_to_new_tensors:
            tensor_maps[i].tensorIndex = map_old_to_new_tensors[
                tensor_maps[i].tensorIndex
            ]


def _remove_tensors_from_model(model, remove_tensors_idxs):
    """Remove tensors from model."""
    if not remove_tensors_idxs:
        return
    if len(model.subgraphs) > 1:
        raise ValueError(
            "Model must only have one subgraph. Instead, it has "
            "{} subgraphs.".format(len(model.subgraphs))
        )
    subgraph = model.subgraphs[0]
    tensors = subgraph.tensors
    operators = subgraph.operators

    # An optimized check to validate if "remove_tensors_idxs" (eg: [4,5,6]) is an
    # exact subset, with ordering, of "tensors" indices (eg: [0,1,2,3,4,5,6]).
    if min(remove_tensors_idxs) == len(tensors) - len(remove_tensors_idxs):
        del tensors[min(remove_tensors_idxs) :]
    else:
        # Map the old tensor indices to new tensor indices
        d_old_to_new_tensors = {}
        left_shift_by = 0
        for idx in range(len(tensors)):
            if idx in remove_tensors_idxs:
                left_shift_by += 1
            else:
                d_old_to_new_tensors[idx] = idx - left_shift_by
        # Update tensor indices referenced throughout the model
        def update_tensors(tensor_idxs):
            for i, ti in enumerate(tensor_idxs):
                tensor_idxs[i] = d_old_to_new_tensors.get(ti, -1)

        update_tensors(subgraph.inputs)
        update_tensors(subgraph.outputs)
        for op in operators:
            update_tensors(op.inputs)
            update_tensors(op.outputs)
        if model.signatureDefs:
            signature_def = model.signatureDefs[0]
            _update_signature_def_tensors(signature_def.inputs, d_old_to_new_tensors)
            _update_signature_def_tensors(signature_def.outputs, d_old_to_new_tensors)
        # Delete the tensors
        for idx in sorted(remove_tensors_idxs, reverse=True):
            tensors.pop(idx)


def _find_int8_quantized_inputs_outputs(model):
    """Validate that model input is quantized and output is dequantized."""
    if len(model.subgraphs) > 1:
        raise ValueError(
            "Model must only have one subgraph. Instead, it has "
            "{} subgraphs.".format(len(model.subgraphs))
        )
    subgraph = model.subgraphs[0]
    tensors = subgraph.tensors
    operators = subgraph.operators

    # Ensure model has atleast one quantize and dequantize operator
    quant_opcode_idx, dequant_opcode_idx = None, None
    for idx, opcode in enumerate(model.operatorCodes):
        if opcode.builtinCode == tflite_schema.BuiltinOperator.QUANTIZE:
            quant_opcode_idx = idx
        elif opcode.builtinCode == tflite_schema.BuiltinOperator.DEQUANTIZE:
            dequant_opcode_idx = idx
        if quant_opcode_idx is not None and dequant_opcode_idx is not None:
            break
    if quant_opcode_idx is None and dequant_opcode_idx is None:
        raise ValueError(
            "Model is not integer quantized as it does not "
            "contain quantize/dequantize operators."
        )

    # Ensure model inputs and outputs are integer quantized
    input_quant_ops, output_dequant_ops = [], []
    for op in operators:
        # Find input quantize operator
        if op.opcodeIndex == quant_opcode_idx and op.inputs[0] in subgraph.inputs:
            pos, float_tensor, int_tensor = (
                "input",
                tensors[op.inputs[0]],
                tensors[op.outputs[0]],
            )
            input_quant_ops.append(op)
        # Find output dequantize operator
        elif op.opcodeIndex == dequant_opcode_idx and op.outputs[0] in subgraph.outputs:
            pos, float_tensor, int_tensor = (
                "output",
                tensors[op.outputs[0]],
                tensors[op.inputs[0]],
            )
            output_dequant_ops.append(op)
        # Otherwise, ignore
        else:
            continue
        # If found, validate the input/output tensor type
        if float_tensor.type != tflite_schema.TensorType.FLOAT32:
            raise ValueError(
                "Model {} type must be tf.float32. Expected type for tensor with "
                "name '{}' is tf.float32, instead type is tf.{}".format(
                    pos,
                    float_tensor.name,
                    _convert_tflite_enum_type_to_tf_type(float_tensor.type).name,
                )
            )
        if int_tensor.type != tflite_schema.TensorType.INT8:
            raise ValueError(
                "Model is not integer quantized. Expected type for tensor with "
                "name '{}' is tf.int8, instead type is tf.{}".format(
                    int_tensor.name,
                    _convert_tflite_enum_type_to_tf_type(int_tensor.type).name,
                )
            )

    return input_quant_ops, output_dequant_ops


def modify_integer_quantized_model_io_type(
    model,
    inference_input_type=tf.float32,
    inference_output_type=tf.float32,
):
    """Modify the float input/output type of an integer quantized model."""
    # Convert the model to an object
    model = _convert_model_from_bytearray_to_object(model)

    # Validate the integer quantized model
    input_quant_ops, output_dequant_ops = _find_int8_quantized_inputs_outputs(model)

    subgraph = model.subgraphs[0]
    operators = subgraph.operators
    remove_tensors_idxs = set()

    # Modify model input type
    if inference_input_type == tf.int8:
        # Remove the inputs and the quant operator
        for op in input_quant_ops:
            subgraph.inputs[subgraph.inputs == op.inputs[0]] = op.outputs[0]
            if model.signatureDefs:
                signature_def = model.signatureDefs[0]
                for i in range(len(signature_def.inputs)):
                    if signature_def.inputs[i].tensorIndex == op.inputs[0]:
                        signature_def.inputs[i].tensorIndex = op.outputs[0]
            remove_tensors_idxs.add(op.inputs[0])
            operators.remove(op)

    # Modify model output type
    if inference_output_type == tf.int8:
        # Remove the outputs and the dequant operator
        for op in output_dequant_ops:
            subgraph.outputs[subgraph.outputs == op.outputs[0]] = op.inputs[0]
            if model.signatureDefs:
                signature_def = model.signatureDefs[0]
                for i in range(len(signature_def.outputs)):
                    if signature_def.outputs[i].tensorIndex == op.outputs[0]:
                        signature_def.outputs[i].tensorIndex = op.inputs[0]
            remove_tensors_idxs.add(op.outputs[0])
            operators.remove(op)

    # Remove tensors marked for deletion.
    _remove_tensors_from_model(model, remove_tensors_idxs)

    # Convert the model to a bytearray
    return _convert_model_from_object_to_bytearray(model)


def strip_lcedequantize_ops(model):
    """Strip the LceDequantize ops to directly output bitpacked tf.int32 tensors."""
    # Convert the model to an object
    model = _convert_model_from_bytearray_to_object(model)

    if len(model.subgraphs) > 1:
        raise ValueError(
            "Model must only have one subgraph. Instead, it has "
            "{} subgraphs.".format(len(model.subgraphs))
        )

    # Ensure model has at least one LceDequantize and/or Dequantize operator
    lce_dequant_opcode_idx, dequant_opcode_idx = None, None
    for idx, opcode in enumerate(model.operatorCodes):
        if opcode.customCode == b"LceDequantize":
            lce_dequant_opcode_idx = idx
        elif opcode.builtinCode == tflite_schema.BuiltinOperator.DEQUANTIZE:
            dequant_opcode_idx = idx
        if lce_dequant_opcode_idx is not None and dequant_opcode_idx is not None:
            break
    if lce_dequant_opcode_idx is None and dequant_opcode_idx is None:
        raise ValueError(
            "Model does not contain any LceDequantize or Dequantize operators."
        )

    # Ensure model outputs are dequantized and remove Dequantize ops first if any
    if dequant_opcode_idx is not None:
        subgraph = model.subgraphs[0]
        tensors = subgraph.tensors
        operators = subgraph.operators
        remove_tensors_idxs = set()

        output_dequant_ops = []
        for op in operators:
            # Find output Dequantize operator
            if (
                op.opcodeIndex == dequant_opcode_idx
                and op.outputs[0] in subgraph.outputs
            ):
                pos, float_tensor, int_tensor = (
                    "output",
                    tensors[op.outputs[0]],
                    tensors[op.inputs[0]],
                )
                output_dequant_ops.append(op)
            # Otherwise, ignore
            else:
                continue
            # If found, validate the input/output tensor type
            if float_tensor.type != tflite_schema.TensorType.FLOAT32:
                raise ValueError(
                    "Model {} type must be tf.float32. Expected type for tensor with "
                    "name '{}' is tf.float32, instead type is tf.{}".format(
                        pos,
                        float_tensor.name,
                        _convert_tflite_enum_type_to_tf_type(float_tensor.type).name,
                    )
                )
            if int_tensor.type != tflite_schema.TensorType.INT8:
                raise ValueError(
                    "Model is not integer quantized. Expected type for tensor with "
                    "name '{}' is tf.int8, instead type is tf.{}".format(
                        int_tensor.name,
                        _convert_tflite_enum_type_to_tf_type(int_tensor.type).name,
                    )
                )

        # Remove the Dequantize operators
        for op in output_dequant_ops:
            subgraph.outputs[subgraph.outputs == op.outputs[0]] = op.inputs[0]
            if model.signatureDefs:
                signature_def = model.signatureDefs[0]
                for i in range(len(signature_def.outputs)):
                    if signature_def.outputs[i].tensorIndex == op.outputs[0]:
                        signature_def.outputs[i].tensorIndex = op.inputs[0]
            remove_tensors_idxs.add(op.outputs[0])
            operators.remove(op)

        # Remove tensors marked for deletion.
        _remove_tensors_from_model(model, remove_tensors_idxs)

    subgraph = model.subgraphs[0]
    tensors = subgraph.tensors
    operators = subgraph.operators
    remove_tensors_idxs = set()

    # Ensure model outputs are Lce dequantized and remove LceDequantize ops
    lce_output_dequant_ops = []
    for op in operators:
        # Find output LceDequantize operator
        if (
            op.opcodeIndex == lce_dequant_opcode_idx
            and op.outputs[0] in subgraph.outputs
        ):
            pos, output_tensor, input_tensor = (
                "output",
                tensors[op.outputs[0]],
                tensors[op.inputs[0]],
            )
            lce_output_dequant_ops.append(op)
        # Otherwise, ignore
        else:
            continue
        # If found, validate the input/output tensor type
        if (
            output_tensor.type != tflite_schema.TensorType.FLOAT32
            and output_tensor.type != tflite_schema.TensorType.INT8
        ):
            raise ValueError(
                "Model {} type must be tf.float32/tf.int8. Expected type for tensor with "
                "name '{}' is tf.float32/tf.int8, instead type is tf.{}".format(
                    pos,
                    output_tensor.name,
                    _convert_tflite_enum_type_to_tf_type(output_tensor.type).name,
                )
            )
        if input_tensor.type != tflite_schema.TensorType.INT32:
            raise ValueError(
                "Expected type for tensor with "
                "name '{}' is tf.int32, instead type is tf.{}".format(
                    input_tensor.name,
                    _convert_tflite_enum_type_to_tf_type(input_tensor.type).name,
                )
            )

    # Remove the LceDequantize operators
    for op in lce_output_dequant_ops:
        subgraph.outputs[subgraph.outputs == op.outputs[0]] = op.inputs[0]
        if model.signatureDefs:
            signature_def = model.signatureDefs[0]
            for i in range(len(signature_def.outputs)):
                if signature_def.outputs[i].tensorIndex == op.outputs[0]:
                    signature_def.outputs[i].tensorIndex = op.inputs[0]
        remove_tensors_idxs.add(op.outputs[0])
        operators.remove(op)

    # Remove tensors marked for deletion.
    _remove_tensors_from_model(model, remove_tensors_idxs)

    # Convert the model to a bytearray
    return _convert_model_from_object_to_bytearray(model)
