#include "larq_compute_engine/mlir/tf_to_tfl_flatbuffer.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
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
    module.emitError(
        "The graph has Control Flow V1 ops. TFLite converter doesn't support "
        "Control Flow V1 ops. Consider using Control Flow V2 ops instead. See "
        "https://www.tensorflow.org/api_docs/python/tf/compat/v1/"
        "enable_control_flow_v2.");
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

Status ConvertTFExecutorToFlatbuffer(mlir::ModuleOp module, bool export_to_mlir,
                                     std::string* result,
                                     mlir::PassManager* pass_manager) {
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

  if (failed(IsValidGraph(module)) || failed(pass_manager->run(module))) {
    return statusHandler.ConsumeStatus();
  }

  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return Status::OK();
  }

  // This is the only modification compared to the upstream tensorflow file
  TruncateOpOrArgLocNameMapper op_or_arg_name_mapper;
  tflite::FlatbufferExportOptions options;
  options.emit_builtin_tflite_ops = true;
  options.emit_custom_ops = true;
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
