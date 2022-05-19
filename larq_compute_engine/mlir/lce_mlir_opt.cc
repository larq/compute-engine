#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

int main(int argc, char** argv) {
  mlir::registerTransformsPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::quant::QuantizationDialect, mlir::TF::TensorFlowDialect,
                  mlir::TFL::TensorFlowLiteDialect, mlir::lq::LarqDialect>();
  return failed(mlir::MlirOptMain(
      argc, argv, "Larq Compute Engine pass driver\n", registry));
}
