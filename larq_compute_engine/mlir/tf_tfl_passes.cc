#include "larq_compute_engine/mlir/tf_tfl_passes.h"

#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OpPassBase<FuncOp>>
CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {

void AddTFToLCETFLConversionPasses(mlir::OpPassManager* pass_manager) {
  pass_manager->addPass(mlir::tf_executor::CreateSwitchFoldPass());
  pass_manager->addPass(mlir::CreateTFExecutorToControlDialectConversion());
  pass_manager->addPass(mlir::TFControlFlow::CreateRaiseTFControlFlowPass());

  // TODO(jpienaar): Revise post dialect constants.
  pass_manager->addPass(mlir::TF::CreateDecodeConstantPass());
  // Canonicalization includes const folding, which is utilized here to optimize
  // away ops that can't get constant folded after PrepareTF pass. For example,
  // tf.Conv2D is split into tf.Transpose and tfl.Conv2D.
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());

  // The below passes only make sense if Builtin TFLite ops are enabled
  // for emission.

  // Inject Larq Compute Engine Ops
  pass_manager->addPass(mlir::TFL::CreatePrepareLCEPass());
  // Prepare for TFLite dialect, rerun canonicalization, and then legalize to
  // the TFLite dialect.
  pass_manager->addPass(mlir::TFL::CreatePrepareTFPass());
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pass_manager->addPass(mlir::TFL::CreateLegalizeTFPass());
  pass_manager->addPass(mlir::TFL::CreateOptimizePass());
  pass_manager->addPass(mlir::TFL::CreateOptimizeLCEPass());
  // This pass operates on TensorFlow ops but is triggered after legalization
  // so that it can target constants introduced once TensorFlow Identity ops
  // are removed during legalization.
  pass_manager->addPass(mlir::TFL::CreateOptimizeFunctionalOpsPass());
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  // This pass should be always at the end of the floating point model
  // conversion. Some TFL ops like unidirectional
  // sequence lstm will have stateful operands and some optimization passes
  // will merge those operands if they have identical values & types. However,
  // it's not desired by TFL. This pass serves as a "fix" pass to split the
  // merged inputs until we have 1st class variable support or reuse
  // tf.variable to model this.
  pass_manager->addPass(mlir::TFL::CreateSplitMergedOperandsPass());
}

}  // namespace tensorflow
