#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

int main(int argc, char** argv) {
  mlir::registerTransformsPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::StandardOpsDialect, mlir::quant::QuantizationDialect,
                  mlir::TF::TensorFlowDialect, mlir::TFL::TensorFlowLiteDialect,
                  mlir::lq::LarqDialect>();
  return failed(mlir::MlirOptMain(
      argc, argv, "Larq Compute Engine pass driver\n", registry));
}
