#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "larq_compute_engine/tflite/python/interpreter_wrapper_utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

class LiteInterpreterWrapper
    : public InterpreterWrapperBase<tflite::Interpreter> {
 public:
  LiteInterpreterWrapper(const pybind11::bytes& flatbuffer,
                         const int num_threads);
  ~LiteInterpreterWrapper(){};

 private:
  std::string flatbuffer_;  // Copy of the flatbuffer because the pybind version
                            // is destroyed

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
};

LiteInterpreterWrapper::LiteInterpreterWrapper(
    const pybind11::bytes& flatbuffer, const int num_threads = 1) {
  // Make a copy of the flatbuffer because it can get deallocated after the
  // constructor is done
  flatbuffer_ = static_cast<std::string>(flatbuffer);

  model_ = tflite::FlatBufferModel::BuildFromBuffer(flatbuffer_.data(),
                                                    flatbuffer_.size());
  if (!model_) {
    PY_ERROR("Invalid model.");
  }

  // Build the interpreter
  resolver_ = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  compute_engine::tflite::RegisterLCECustomOps(resolver_.get());

  tflite::InterpreterBuilder builder(*model_, *resolver_);
  builder(&interpreter_, num_threads);
  MINIMAL_CHECK(interpreter_ != nullptr);

  // Allocate tensor buffers.
  MINIMAL_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);
}

PYBIND11_MODULE(interpreter_wrapper_lite, m) {
  pybind11::class_<LiteInterpreterWrapper>(m, "LiteInterpreter")
      .def(pybind11::init<const pybind11::bytes&, const int>())
      .def_property("input_types", &LiteInterpreterWrapper::get_input_types,
                    nullptr)
      .def_property("output_types", &LiteInterpreterWrapper::get_output_types,
                    nullptr)
      .def_property("input_shapes", &LiteInterpreterWrapper::get_input_shapes,
                    nullptr)
      .def_property("output_shapes", &LiteInterpreterWrapper::get_output_shapes,
                    nullptr)
      .def("predict", &LiteInterpreterWrapper::predict);
};
