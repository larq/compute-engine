#include "larq_compute_engine/mlir/tf_tfl_passes.h"

#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OperationPass<FuncOp>>
CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {

void AddQuantizationPasses(const mlir::TFL::QuantizationSpecs& quant_specs,
                           mlir::OpPassManager* pass_manager) {
  pass_manager->addPass(mlir::TFL::CreatePrepareQuantizePass(quant_specs));
  pass_manager->addPass(mlir::TFL::CreateHybridQuantizePass());
  bool emit_quant_adaptor_ops =
      quant_specs.inference_type != quant_specs.inference_input_type;
  pass_manager->addPass(
      mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));

  if (quant_specs.default_ranges.first.hasValue() ||
      quant_specs.default_ranges.second.hasValue()) {
    pass_manager->addPass(mlir::TFL::CreateDefaultQuantParamsPass(
        quant_specs.default_ranges.first.getValueOr(0.0),
        quant_specs.default_ranges.second.getValueOr(0.0),
        quant_specs.IsSignedInferenceType()));
    pass_manager->addPass(mlir::TFL::CreateHybridQuantizePass());
    pass_manager->addPass(
        mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
  }
}

void AddTFToLCETFLConversionPasses(
    const mlir::TFL::QuantizationSpecs& quant_specs,
    mlir::OpPassManager* pass_manager,
    bool experimental_enable_bitpacked_activations) {
  pass_manager->addPass(mlir::tf_executor::CreateSwitchFoldPass());
  pass_manager->addPass(mlir::CreateTFExecutorToControlDialectConversion());
  pass_manager->addPass(mlir::TFControlFlow::CreateRaiseTFControlFlowPass());

  // The conversion pipeline has to follow the following orders:
  // 1) Saved model related optimization like decompose resource ops
  // 2) Convert composite functions like lstm/rnns, along with proper function
  // inlining & dce.
  // 3) Lower static tensor list pass.

  // This decomposes resource ops like ResourceGather into read-variable op
  // followed by gather. This is used when the saved model import path is used
  // during which resources dont get frozen in the python layer.
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());

  // Enable fusing composite ops that can be lowered to built-in TFLite ops.
  pass_manager->addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());

  // This pass marks non-exported functions as symbol visibility 'private'
  // those deemed read-only as immutable.
  pass_manager->addPass(
      mlir::tf_saved_model::
          CreateMarkFunctionVisibilityUsingSavedModelLinkagePass());

  pass_manager->addPass(mlir::createInlinerPass());
  pass_manager->addPass(mlir::createSymbolDCEPass());

  pass_manager->addPass(mlir::TFL::CreateLowerStaticTensorListPass());

  // This pass does resource analysis of saved model global tensors and marks
  // those deemed read-only as immutable.
  pass_manager->addPass(
      mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());

  // Add a shape inference pass to optimize away the unnecessary casts.
  pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());

  // Legalize while early to allow further constant folding.
  // TODO(jpienaar): This may not actually matter as we do canonicalization
  // after the legalize below, for now it needs to be below the above passes
  // that work on TF dialect and before inliner so that the function calls in
  // body and cond are inlined for optimization.
  pass_manager->addPass(mlir::TFL::CreateLegalizeTFWhilePass());

  // Add function inlining pass. Both TF and TFLite dialects are opted into
  // function inliner interface.
  pass_manager->addPass(mlir::createInlinerPass());

  // TODO(jpienaar): Revise post dialect constants.
  pass_manager->addPass(mlir::TF::CreateDecodeConstantPass());
  // Remove passthrough ops early so constant folding can happen before
  // LCE ops are injected
  pass_manager->addPass(mlir::TFL::CreateOpRemovalPass());
  // Canonicalization includes const folding, which is utilized here to optimize
  // away ops that can't get constant folded after PrepareTF pass. For example,
  // tf.Conv2D is split into tf.Transpose and tfl.Conv2D.
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  // This pass does dead code elimination based on symbol visibility.
  pass_manager->addPass(mlir::createSymbolDCEPass());

  // This pass 'freezes' immutable global tensors and inlines them as tf
  // constant ops.
  pass_manager->addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass());

  // Inject Larq Compute Engine Ops
  pass_manager->addPass(mlir::TFL::CreatePrepareLCEPass());
  // Prepare for TFLite dialect, rerun canonicalization, and then legalize to
  // the TFLite dialect.
  pass_manager->addPass(mlir::TFL::CreatePrepareTFPass(true));
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pass_manager->addPass(mlir::TFL::CreateLegalizeTFPass(true));
  pass_manager->addPass(mlir::TFL::CreateOptimizePass());
  pass_manager->addPass(mlir::TFL::CreateOptimizeLCEPass(
      experimental_enable_bitpacked_activations));
  pass_manager->addPass(mlir::TFL::CreateBitpackWeightsLCEPass());
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

  // Run quantization after all the floating point model conversion is
  // completed.
  if (quant_specs.RunPropagationAndRewriteQuantizationPasses()) {
    AddQuantizationPasses(quant_specs, pass_manager);
  }
}

}  // namespace tensorflow
