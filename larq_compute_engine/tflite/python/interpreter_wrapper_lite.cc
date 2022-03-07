#include <memory>

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
                         const int num_threads = 1,
                         const bool use_reference_bconv = false
                         const bool use_indirect_bgemm = false
                         const bool use_xnnpack = false);
  ~LiteInterpreterWrapper(){};

 private:
  std::string flatbuffer_;  // Copy of the flatbuffer because the pybind version
                            // is destroyed

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
};

LiteInterpreterWrapper::LiteInterpreterWrapper(
    const pybind11::bytes& flatbuffer, const int num_threads,
    const bool use_reference_bconv, const bool use_indirect_bgemm, const bool use_xnnpack) {
  // Make a copy of the flatbuffer because it can get deallocated after the
  // constructor is done
  flatbuffer_ = static_cast<std::string>(flatbuffer);

  model_ = tflite::FlatBufferModel::BuildFromBuffer(flatbuffer_.data(),
                                                    flatbuffer_.size());
  if (!model_) {
    PY_ERROR("Invalid model.");
  }

  // Build the interpreter
  if (use_xnnpack) {
    resolver_ = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  } else {
    resolver_ = std::make_unique<tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>();
  }
  compute_engine::tflite::RegisterLCECustomOps(resolver_.get(),
                                               use_reference_bconv, use_indirect_bgemm);

  tflite::InterpreterBuilder builder(*model_, *resolver_);
  builder(&interpreter_, num_threads);
  MINIMAL_CHECK(interpreter_ != nullptr);

  // Allocate tensor buffers.
  MINIMAL_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);
}

PYBIND11_MODULE(interpreter_wrapper_lite, m) {
  pybind11::class_<LiteInterpreterWrapper>(m, "LiteInterpreter")
      .def(pybind11::init<const pybind11::bytes&, const int, const bool,
           const bool, const bool>())
      .def_property("input_types", &LiteInterpreterWrapper::get_input_types,
                    nullptr)
      .def_property("output_types", &LiteInterpreterWrapper::get_output_types,
                    nullptr)
      .def_property("input_shapes", &LiteInterpreterWrapper::get_input_shapes,
                    nullptr)
      .def_property("output_shapes", &LiteInterpreterWrapper::get_output_shapes,
                    nullptr)
      .def_property("input_zero_points",
                    &LiteInterpreterWrapper::get_input_zero_points, nullptr)
      .def_property("output_zero_points",
                    &LiteInterpreterWrapper::get_output_zero_points, nullptr)
      .def_property("input_scales", &LiteInterpreterWrapper::get_input_scales,
                    nullptr)
      .def_property("output_scales", &LiteInterpreterWrapper::get_output_scales,
                    nullptr)
      .def("predict", &LiteInterpreterWrapper::predict);
};
