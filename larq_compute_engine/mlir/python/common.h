#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "pybind11/pybind11.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

LCETarget GetLCETarget(const std::string& target_str);

Status GetNumInputs(mlir::OwningModuleRef* module, int* num_inputs);

pybind11::bytes ConvertMLIRModuleToTFLiteFlatBuffer(
    mlir::OwningModuleRef* module, mlir::MLIRContext& context,
    const LCETarget target, const pybind11::object& default_ranges,
    const int num_inputs, const bool should_quantize,
    const bool mark_as_post_training_quant);

}  // namespace tensorflow
