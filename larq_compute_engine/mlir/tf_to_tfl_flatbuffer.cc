#include "larq_compute_engine/mlir/tf_to_tfl_flatbuffer.h"

#include "larq_compute_engine/mlir/tf_tfl_passes.h"
#include "larq_compute_engine/mlir/transforms/passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {
using mlir::ModuleOp;
using mlir::Operation;

bool IsControlFlowV1Op(Operation* op) {
  return mlir::isa<mlir::tf_executor::SwitchOp, mlir::tf_executor::MergeOp,
                   mlir::tf_executor::EnterOp, mlir::tf_executor::ExitOp,
                   mlir::tf_executor::NextIterationSinkOp,
                   mlir::tf_executor::NextIterationSourceOp>(op);
}

mlir::LogicalResult IsValidGraph(mlir::ModuleOp module) {
  auto result = module.walk([&](Operation* op) {
    return IsControlFlowV1Op(op) ? mlir::WalkResult::interrupt()
                                 : mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    mlir::TFL::AttachErrorCode(
        module.emitError(
            "The graph has Control Flow V1 ops. TFLite converter doesn't "
            "support Control Flow V1 ops. Consider using Control Flow V2 ops "
            "instead. See https://www.tensorflow.org/api_docs/python/tf/compat/"
            "v1/enable_control_flow_v2."),
        tflite::metrics::ConverterErrorData::ERROR_UNSUPPORTED_CONTROL_FLOW_V1);
    return mlir::failure();
  }
  return mlir::success();
}

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

}  // namespace
Status ConvertTFExecutorToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir, const LCETarget target,
    mlir::TFL::QuantizationSpecs quant_specs,
    const std::unordered_set<std::string>& saved_model_tags,
    llvm::StringRef saved_model_dir,
    llvm::Optional<tensorflow::Session*> session, std::string* result) {
  // Explicitly disable dumping Op details on failures.
  module.getContext()->printOpOnDiagnostic(false);

  // Register a warning handler only log to std out.
  mlir::ScopedDiagnosticHandler s(
      module.getContext(), [](mlir::Diagnostic& diag) {
        if (diag.getSeverity() == mlir::DiagnosticSeverity::Warning) {
          for (auto& note : diag.getNotes()) {
            std::cout << note.str() << "\n";
            LOG(WARNING) << note.str() << "\n";
          }
        }
        return mlir::failure();
      });

  mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext(),
                                                    /*propagate=*/true);
  if (failed(IsValidGraph(module))) {
    return statusHandler.ConsumeStatus();
  }

  mlir::PassManager pass_manager(module.getContext());
  mlir::applyPassManagerCLOptions(pass_manager);
  pass_manager.addInstrumentation(
      std::make_unique<mlir::TFL::ErrorCollectorInstrumentation>(
          pass_manager.getContext()));

  tensorflow::AddPreVariableFreezingTFToLCETFLConversionPasses(&pass_manager);
  if (failed(pass_manager.run(module))) {
    return statusHandler.ConsumeStatus();
  }

  // Freeze variables if a session is provided.
  if (session.hasValue()) {
    mlir::TFL::ErrorCollectorInstrumentation collector(module.getContext());
    if (failed(mlir::tf_saved_model::FreezeVariables(module,
                                                     session.getValue()))) {
      auto status = statusHandler.ConsumeStatus();
      mlir::TFL::ErrorCollector* collector =
          mlir::TFL::ErrorCollector::GetErrorCollector();
      if (!collector->CollectedErrors().empty()) {
        return errors::InvalidArgument("Variable constant folding is failed.");
      }
      return status;
    }
  }
  pass_manager.clear();
  tensorflow::AddPostVariableFreezingTFToLCETFLConversionPasses(
      saved_model_dir, quant_specs, &pass_manager, target);
  if (failed(pass_manager.run(module))) {
    auto status = statusHandler.ConsumeStatus();
    mlir::TFL::ErrorCollector* collector =
        mlir::TFL::ErrorCollector::GetErrorCollector();
    for (const auto& error_data : collector->CollectedErrors()) {
      if (error_data.subcomponent() == "FreezeGlobalTensorsPass") {
        return errors::InvalidArgument("Variable constant folding is failed.");
      }
    }
    return status;
  }

  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return statusHandler.ConsumeStatus();
  }

  // Write MLIR TFLite dialect into FlatBuffer
  TruncateOpOrArgLocNameMapper op_or_arg_name_mapper;
  toco::TocoFlags toco_flags;
  toco_flags.set_force_select_tf_ops(false);
  toco_flags.set_allow_custom_ops(true);
  tflite::FlatbufferExportOptions options;
  options.toco_flags = toco_flags;
  options.saved_model_tags = saved_model_tags;
  options.op_or_arg_name_mapper = &op_or_arg_name_mapper;
  if (!tflite::MlirToFlatBufferTranslateFunction(module, options, result)) {
    return statusHandler.ConsumeStatus();
  }

  if (mlir::failed(module.verify())) {
    return tensorflow::errors::Unknown("Final module is invalid");
  }
  return Status::OK();
}

}  // namespace tensorflow
