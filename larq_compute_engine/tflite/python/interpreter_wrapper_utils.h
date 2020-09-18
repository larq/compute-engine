#ifndef COMPUTE_ENGINE_TFLITE_PYTHON_INTERPRETER_WRAPPER_UTILS_H_
#define COMPUTE_ENGINE_TFLITE_PYTHON_INTERPRETER_WRAPPER_UTILS_H_

#include <cstring>
#include <sstream>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/lite/c/common.h"

#define PY_ERROR(x)                                                \
  {                                                                \
    std::stringstream ss;                                          \
    ss << "ERROR at " << __FILE__ << ":" << __LINE__ << " : " << x \
       << std::endl;                                               \
    throw std::runtime_error(ss.str());                            \
  }

#define MINIMAL_CHECK(x)                        \
  if (!(x)) {                                   \
    PY_ERROR("the following was false: " << #x) \
  }

template <typename InterpreterType>
class InterpreterWrapperBase {
 public:
  InterpreterWrapperBase(){};
  ~InterpreterWrapperBase(){};

  // The python object `input` is a `List` of numpy arrays,
  // one numpy array for each model input.
  //
  // The result is a `List` of numpy arrays, one for each output
  pybind11::list predict(const pybind11::list& input_list);

  // List of numpy types
  pybind11::list get_input_types() const;
  pybind11::list get_output_types() const;
  // List of shape tuples
  pybind11::list get_input_shapes() const;
  pybind11::list get_output_shapes() const;

 protected:
  std::unique_ptr<InterpreterType> interpreter_;
};

TfLiteType TfLiteTypeFromPyType(pybind11::dtype py_type) {
  if (py_type.is(pybind11::dtype::of<float>())) return kTfLiteFloat32;
  if (py_type.is(pybind11::dtype::of<std::uint8_t>())) return kTfLiteUInt8;
  if (py_type.is(pybind11::dtype::of<std::int8_t>())) return kTfLiteInt8;
  if (py_type.is(pybind11::dtype::of<std::int16_t>())) return kTfLiteInt16;
  if (py_type.is(pybind11::dtype::of<std::int32_t>())) return kTfLiteInt32;
  if (py_type.is(pybind11::dtype::of<std::int64_t>())) return kTfLiteInt64;
  if (py_type.is(pybind11::dtype::of<bool>())) return kTfLiteBool;
  if (py_type.is(pybind11::dtype::of<char*>())) return kTfLiteString;
  return kTfLiteNoType;
}

pybind11::dtype PyTypeFromTfLiteType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return pybind11::dtype::of<float>();
    case kTfLiteUInt8:
      return pybind11::dtype::of<std::uint8_t>();
    case kTfLiteInt8:
      return pybind11::dtype::of<std::int8_t>();
    case kTfLiteInt16:
      return pybind11::dtype::of<std::int16_t>();
    case kTfLiteInt32:
      return pybind11::dtype::of<std::int32_t>();
    case kTfLiteInt64:
      return pybind11::dtype::of<std::int64_t>();
    case kTfLiteBool:
      return pybind11::dtype::of<bool>();
    case kTfLiteString:
      return pybind11::dtype::of<char*>();
    case kTfLiteNoType:
    default:
      PY_ERROR("Model has invalid output type: " << type);
      return pybind11::dtype::of<float>();
  };
}

bool SetTensorFromNumpy(const TfLiteTensor* tensor,
                        const pybind11::array& nparray) {
  TfLiteType type = TfLiteTypeFromPyType(nparray.dtype());

  if (type != tensor->type) {
    PY_ERROR("Expected tensor type " << TfLiteTypeGetName(tensor->type)
                                     << " but got " << TfLiteTypeGetName(type)
                                     << " for tensor " << tensor->name);
    return false;
  }

  int ndim = nparray.ndim();
  if (ndim != tensor->dims->size) {
    PY_ERROR("Expected tensor dimension " << tensor->dims->size << " but got "
                                          << ndim << " for tensor "
                                          << tensor->name);
    return false;
  }

  for (int j = 0; j < ndim; j++) {
    if (tensor->dims->data[j] != nparray.shape(j)) {
      PY_ERROR("Expected " << tensor->dims->data[j] << " for dimension " << j
                           << " but found " << nparray.shape(j)
                           << " for tensor " << tensor->name);
      return false;
    }
  }

  size_t size = nparray.nbytes();
  if (size != tensor->bytes) {
    PY_ERROR("Expected " << tensor->bytes << " bytes but got " << size
                         << " for tensor " << tensor->name);
    return false;
  }
  memcpy(tensor->data.raw, nparray.data(), size);
  return true;
}

template <typename InterpreterType>
pybind11::list InterpreterWrapperBase<InterpreterType>::get_input_types()
    const {
  MINIMAL_CHECK(interpreter_);
  pybind11::list result;

  for (auto tensor_id : interpreter_->inputs()) {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_id);
    result.append(PyTypeFromTfLiteType(tensor->type));
  }

  return result;
}

template <typename InterpreterType>
pybind11::list InterpreterWrapperBase<InterpreterType>::get_output_types()
    const {
  MINIMAL_CHECK(interpreter_);
  pybind11::list result;

  for (auto tensor_id : interpreter_->outputs()) {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_id);
    result.append(PyTypeFromTfLiteType(tensor->type));
  }

  return result;
}

template <typename InterpreterType>
pybind11::list InterpreterWrapperBase<InterpreterType>::get_input_shapes()
    const {
  MINIMAL_CHECK(interpreter_);
  pybind11::list result;

  for (auto tensor_id : interpreter_->inputs()) {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_id);
    pybind11::tuple shape(tensor->dims->size);
    for (int j = 0; j < tensor->dims->size; ++j)
      shape[j] = tensor->dims->data[j];
    result.append(shape);
  }

  return result;
}

template <typename InterpreterType>
pybind11::list InterpreterWrapperBase<InterpreterType>::get_output_shapes()
    const {
  MINIMAL_CHECK(interpreter_);
  pybind11::list result;

  for (auto tensor_id : interpreter_->outputs()) {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_id);
    pybind11::tuple shape(tensor->dims->size);
    for (int j = 0; j < tensor->dims->size; ++j)
      shape[j] = tensor->dims->data[j];
    result.append(shape);
  }

  return result;
}

template <typename InterpreterType>
pybind11::list InterpreterWrapperBase<InterpreterType>::predict(
    const pybind11::list& input_list) {
  MINIMAL_CHECK(interpreter_);
  const size_t inputs_size = input_list.size();
  if (inputs_size != interpreter_->inputs().size()) {
    PY_ERROR("Expected " << interpreter_->inputs().size()
                         << " input tensors but got " << inputs_size
                         << " tensors in dataset.");
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    pybind11::array nparray =
        pybind11::array::ensure(input_list[i], pybind11::array::c_style);
    const TfLiteTensor* tensor =
        interpreter_->tensor(interpreter_->inputs()[i]);
    if (!SetTensorFromNumpy(tensor, nparray)) {
      PY_ERROR("Failed to set tensor data of input " << i);
    }
  }

  MINIMAL_CHECK(interpreter_->Invoke() == kTfLiteOk);

  pybind11::list result;
  for (auto output_id : interpreter_->outputs()) {
    TfLiteTensor* tensor = interpreter_->tensor(output_id);
    std::vector<int> shape(tensor->dims->data,
                           tensor->dims->data + tensor->dims->size);
    pybind11::array nparray(PyTypeFromTfLiteType(tensor->type), shape,
                            tensor->data.raw);
    result.append(nparray);
  }

  return result;
}

#endif
