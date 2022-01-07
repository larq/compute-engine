#include <exception>

#include "larq_compute_engine/mlir/python/common.h"
#include "larq_compute_engine/mlir/tf_tfl_passes.h"
#include "larq_compute_engine/mlir/tf_to_tfl_flatbuffer.h"
#include "mlir/IR/MLIRContext.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"

namespace tensorflow {

pybind11::bytes ConvertGraphDefToTFLiteFlatBuffer(
    const pybind11::bytes& graphdef_bytes,
    const std::vector<string>& input_arrays,
    const std::vector<string>& input_dtypes,
    const std::vector<std::vector<int>>& input_shapes,
    const std::vector<string>& output_arrays, const bool should_quantize,
    const std::string& target_str, const pybind11::object& default_ranges) {
  GraphDef graphdef;
  if (!tensorflow::LoadProtoFromBuffer(std::string(graphdef_bytes), &graphdef)
           .ok()) {
    throw std::runtime_error("Could not load GraphDef.");
  }

  auto target = GetLCETarget(target_str);

  // `ParseInputArrayInfo` requires a type that isn't pybind compatible, so
  // translate here.
  std::vector<llvm::Optional<std::vector<int>>> translated_input_shapes;
  for (auto x : input_shapes) {
    if (x.size() > 0) {
      translated_input_shapes.push_back(x);
    } else {
      translated_input_shapes.push_back(llvm::None);
    }
  }

  GraphImportConfig specs;
  specs.prune_unused_nodes = true;
  specs.convert_legacy_fed_inputs = true;
  specs.graph_as_function = false;
  specs.upgrade_legacy = true;
  if (!ParseInputArrayInfo(input_arrays, input_dtypes, translated_input_shapes,
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

  return ConvertMLIRModuleToTFLiteFlatBuffer(
      &module.ValueOrDie(), context, target, default_ranges,
      input_arrays.size(), should_quantize,
      /*mark_as_post_training_quant=*/false);
}

}  // namespace tensorflow
