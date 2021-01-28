#include <exception>

#include "larq_compute_engine/mlir/tf_tfl_passes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

namespace tensorflow {

// Truncates names to a maximum length of ~50 characters since LCE op location
// names can be very long otherwise.
class TruncateOpOrArgLocNameMapper : public OpOrArgLocNameMapper {
 protected:
  std::string GetName(OpOrVal op_or_val) override {
    auto name = OpOrArgLocNameMapper::GetName(op_or_val);
    if (name.length() > 50) return name.substr(0, 50);
    return name;
  }
};

pybind11::bytes ConvertGraphDefToTFLiteFlatBuffer(
    const pybind11::bytes& graphdef_bytes,
    const std::vector<string>& input_arrays,
    const std::vector<string>& input_dtypes,
    const std::vector<std::vector<int>>& input_shapes,
    const std::vector<string>& output_arrays, const bool should_quantize,
    const std::string& target_str, const pybind11::object& default_ranges,
    const bool experimental_enable_bitpacked_activations) {
  GraphDef graphdef;
  if (!tensorflow::LoadProtoFromBuffer(std::string(graphdef_bytes), &graphdef)
           .ok()) {
    throw std::runtime_error("Could not load GraphDef.");
  }

  LCETarget target;
  if (target_str == "arm") {
    target = LCETarget::ARM;
  } else if (target_str == "xcore") {
    target = LCETarget::XCORE;
  } else {
    throw std::runtime_error("Invalid target.");
  }

  GraphImportConfig specs;
  specs.prune_unused_nodes = true;
  specs.convert_legacy_fed_inputs = true;
  specs.graph_as_function = false;
  specs.upgrade_legacy = true;
  if (!ParseInputArrayInfo(input_arrays, input_dtypes, input_shapes,
                           &specs.inputs)
           .ok()) {
    throw std::runtime_error("Could not parse input arrays.");
  }
  if (!ParseOutputArrayInfo(output_arrays, &specs.outputs).ok()) {
    throw std::runtime_error("Could not parse output arrays.");
  }

  mlir::MLIRContext context;
  GraphDebugInfo debug_info;
  mlir::StatusScopedDiagnosticHandler statusHandler(&context,
                                                    /*propagate=*/true);
  auto module = ConvertGraphdefToMlir(graphdef, debug_info, specs, &context);

  if (!module.ok()) {
    throw std::runtime_error("Could not convert GraphDef.");
  }

  mlir::TFL::QuantizationSpecs quant_specs;
  if (should_quantize) {
    quant_specs.inference_type = tensorflow::DT_QINT8;
    for (auto input_array : input_arrays) {
      // Input inference type is DT_FLOAT, so set the default input ranges
      // which default to mean=0.0 and std=1.0.
      quant_specs.input_ranges.push_back({-128.0, 127.0});
    }
    if (!default_ranges.is_none()) {
      quant_specs.default_ranges =
          default_ranges.cast<std::pair<double, double>>();
    }
  }
  mlir::PassManager pm(&context);
  tensorflow::AddTFToLCETFLConversionPasses(
      quant_specs, &pm, target, experimental_enable_bitpacked_activations);

  // Convert back to outlined while format for export back to flatbuffer.
  pm.addPass(mlir::TFL::CreateWhileOutlinePass());
  pm.addPass(mlir::TFL::CreateRuntimeVerifyPass());

  if (failed(pm.run(*module.ValueOrDie()))) {
    throw std::runtime_error("Could not complete conversion passes.");
  }

  TruncateOpOrArgLocNameMapper op_or_arg_name_mapper;
  // Write MLIR TFLite dialect into FlatBuffer
  std::string result;
  if (tflite::MlirToFlatBufferTranslateFunction(
          *module.ValueOrDie(), &result, /*emit_builtin_tflite_ops=*/true,
          /*emit_select_tf_ops=*/false, /*emit_custom_ops=*/true,
          &op_or_arg_name_mapper)) {
    throw std::runtime_error("Could not translate to flatbuffer.");
  }

  return pybind11::bytes(result);
}

}  // namespace tensorflow

PYBIND11_MODULE(_graphdef_tfl_flatbuffer, m) {
  m.def("convert_graphdef_to_tflite_flatbuffer",
        &tensorflow::ConvertGraphDefToTFLiteFlatBuffer);
};
