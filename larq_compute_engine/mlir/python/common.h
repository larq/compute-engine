#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "pybind11/pybind11.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

LCETarget GetLCETarget(const std::string& target_str);

Status GetNumInputs(mlir::OwningOpRef<mlir::ModuleOp>* module, int* num_inputs);

pybind11::bytes ConvertMLIRModuleToTFLiteFlatBuffer(
    mlir::OwningOpRef<mlir::ModuleOp>* module, mlir::MLIRContext& context,
    const LCETarget target, const pybind11::object& default_ranges,
    const std::unordered_set<std::string>& saved_model_tags,
    llvm::StringRef saved_model_dir,
    llvm::Optional<tensorflow::Session*> session, const int num_inputs,
    const bool should_quantize, const bool mark_as_post_training_quant);

}  // namespace tensorflow
