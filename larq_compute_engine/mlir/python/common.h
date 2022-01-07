

#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "pybind11/pybind11.h"

namespace tensorflow {

LCETarget GetLCETarget(const std::string& target_str);

pybind11::bytes ConvertMLIRModuleToTFLiteFlatBuffer(
    mlir::OwningModuleRef* module, mlir::MLIRContext& context,
    const LCETarget target, const pybind11::object& default_ranges);

}  // namespace tensorflow
