#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "larq_compute_engine/mlir/quantization/quantize_model.h"
#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"
#include "tensorflow/lite/tools/optimize/calibration/calibrator.h"
#include "tensorflow/lite/tools/optimize/quantize_model.h"

#define TFLITE_PY_CHECK(x)                                \
  if ((x) != kTfLiteOk) {                                 \
    throw std::runtime_error(error_reporter_->message()); \
  }

#define TFLITE_PY_ENSURE_VALID_INTERPRETER()          \
  if (!interpreter_) {                                \
    throw std::runtime_error("Invalid interpreter."); \
  }

#define PY_ERROR(x)                     \
  {                                     \
    std::stringstream ss;               \
    ss << "ERROR: " << x << std::endl;  \
    throw std::runtime_error(ss.str()); \
  }

namespace py = pybind11;

namespace tflite {
namespace calibration_wrapper {

class CalibrationWrapper {
 public:
  CalibrationWrapper(const py::bytes& flatbuffer);
  ~CalibrationWrapper(){};

  bool Prepare();

  bool FeedTensor(const py::list& input_value);

  py::bytes QuantizeModel(int input_py_type, int output_py_type);

 private:
  bool SetTensor(int index,
                 const py::array_t<float, py::array::c_style>& nparray);

  std::string flatbuffer_;  // Copy of the flatbuffer
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
      error_reporter_;
  std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader_;
};

CalibrationWrapper::CalibrationWrapper(const py::bytes& flatbuffer) {
  using tflite::interpreter_wrapper::PythonErrorReporter;
  error_reporter_ = absl::make_unique<PythonErrorReporter>();
  ::tflite::python::ImportNumpy();

  // Make a copy of the flatbuffer because it can get deallocated after the
  // constructor is done
  flatbuffer_ = static_cast<std::string>(flatbuffer);

  model_ = tflite::FlatBufferModel::BuildFromBuffer(
      flatbuffer_.data(), flatbuffer_.size(), error_reporter_.get());

  if (!model_) {
    throw std::runtime_error("Invalid model");
  }

  resolver_ = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();

  compute_engine::tflite::RegisterLCECustomOps(resolver_.get());

  // Note: this will call the `Init` functions of every node
  TFLITE_PY_CHECK(tflite::optimize::calibration::BuildLoggingInterpreter(
      *model_, *resolver_, &interpreter_, &reader_));
}

std::unique_ptr<tflite::ModelT> CreateMutableModel(const tflite::Model& model) {
  std::unique_ptr<tflite::ModelT> copied_model =
      absl::make_unique<tflite::ModelT>();
  model.UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}

inline TensorType TfLiteTypeToSchemaType(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return TensorType_FLOAT32;  // TODO(b/129336260): No schema type for none.
    case kTfLiteFloat32:
      return TensorType_FLOAT32;
    case kTfLiteFloat16:
      return TensorType_FLOAT16;
    case kTfLiteInt32:
      return TensorType_INT32;
    case kTfLiteUInt8:
      return TensorType_UINT8;
    case kTfLiteInt8:
      return TensorType_INT8;
    case kTfLiteInt64:
      return TensorType_INT64;
    case kTfLiteString:
      return TensorType_STRING;
    case kTfLiteBool:
      return TensorType_BOOL;
    case kTfLiteInt16:
      return TensorType_INT16;
    case kTfLiteComplex64:
      return TensorType_COMPLEX64;
  }
  return TensorType_FLOAT32;
}

bool CalibrationWrapper::Prepare() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  // Note: this will call the Prepare function of every node
  TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  return true;
}

bool CalibrationWrapper::FeedTensor(const py::list& input_value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();

  const size_t inputs_size = input_value.size();

  if (inputs_size != interpreter_->inputs().size()) {
    PY_ERROR("Expected " << interpreter_->inputs().size()
                         << " input tensors but got " << inputs_size
                         << " tensors in dataset.");
    return false;
  }

  for (size_t i = 0; i < inputs_size; i++) {
    py::array_t<float, py::array::c_style> input(input_value[i]);
    int input_tensor_idx = interpreter_->inputs()[i];
    if (!SetTensor(input_tensor_idx, input)) {
      PY_ERROR("Failed to set tensor data.");
      return false;
    }
  }

  TFLITE_PY_CHECK(interpreter_->Invoke());

  return true;
}

TfLiteType TfLiteTypeFromPyType(py::dtype py_type) {
  if (py_type.is(py::dtype::of<float>())) return kTfLiteFloat32;
  if (py_type.is(py::dtype::of<std::uint8_t>())) return kTfLiteUInt8;
  if (py_type.is(py::dtype::of<std::int8_t>())) return kTfLiteInt8;
  if (py_type.is(py::dtype::of<std::int16_t>())) return kTfLiteInt16;
  if (py_type.is(py::dtype::of<std::int32_t>())) return kTfLiteInt32;
  if (py_type.is(py::dtype::of<std::int64_t>())) return kTfLiteInt64;
  if (py_type.is(py::dtype::of<bool>())) return kTfLiteBool;
  if (py_type.is(py::dtype::of<char*>())) return kTfLiteString;
  return kTfLiteNoType;
}

bool CalibrationWrapper::SetTensor(
    int index, const py::array_t<float, py::array::c_style>& nparray) {
  const TfLiteTensor* tensor = interpreter_->tensor(index);

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

py::bytes CalibrationWrapper::QuantizeModel(int input_py_type,
                                            int output_py_type) {
  TfLiteType input_type = python_utils::TfLiteTypeFromPyType(input_py_type);
  TfLiteType output_type = python_utils::TfLiteTypeFromPyType(output_py_type);
  if (input_type == kTfLiteNoType || output_type == kTfLiteNoType) {
    PY_ERROR("Input/output type cannot be kTfLiteNoType");
    return nullptr;
  }
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto status = kTfLiteOk;
#if 0
  status = mlir::lite::QuantizeModel(
      *tflite_model, TfLiteTypeToSchemaType(input_type),
      TfLiteTypeToSchemaType(output_type), {}, &builder, error_reporter_.get());
#else
  const bool allow_float = true;
  status = tflite::optimize::QuantizeModel(
      &builder, tflite_model.get(), TfLiteTypeToSchemaType(input_type),
      TfLiteTypeToSchemaType(output_type), allow_float, error_reporter_.get());
#endif

  TFLITE_PY_CHECK(status);

  return pybind11::bytes(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

}  // namespace calibration_wrapper
}  // namespace tflite

using namespace tflite::calibration_wrapper;

PYBIND11_MODULE(_calibration_wrapper, m) {
  py::class_<CalibrationWrapper>(m, "Calibrator")
      .def(py::init<const py::bytes&>())
      .def("Prepare", &CalibrationWrapper::Prepare)
      .def("FeedTensor", &CalibrationWrapper::FeedTensor)
      .def("QuantizeModel", &CalibrationWrapper::QuantizeModel);
};
