#include "larq_compute_engine/mlir/tf_tfl_passes.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_translate_cl.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using mlir::FuncOp;
using mlir::MLIRContext;
using mlir::ModuleOp;
using stream_executor::port::StatusOr;

enum TranslationStatus { kTrSuccess, kTrFailure };

int main(int argc, char** argv) {
  // TODO(jpienaar): Revise the command line option parsing here.
  tensorflow::InitMlir y(&argc, &argv);

  // TODO(antiagainst): We are pulling in multiple transformations as follows.
  // Each transformation has its own set of command-line options; options of one
  // transformation can essentially be aliases to another. For example, the
  // -tfl-annotate-inputs has -tfl-input-arrays, -tfl-input-data-types, and
  // -tfl-input-shapes, which are the same as -graphdef-to-mlir transformation's
  // -tf_input_arrays, -tf_input_data_types, and -tf_input_shapes, respectively.
  // We need to disable duplicated ones to provide a cleaner command-line option
  // interface. That also means we need to relay the value set in one option to
  // all its aliases.
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TF GraphDef to TFLite FlatBuffer converter\n");

  MLIRContext context;
  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(source_mgr, &context);

  StatusOr<mlir::OwningModuleRef> module =
      tensorflow::LoadFromGraphdefOrMlirSource(
          input_file_name, input_mlir, use_splatted_constant, custom_opdefs,
          debug_info_file, input_arrays, input_dtypes, input_shapes,
          output_arrays,
          /*prune_unused_nodes=*/true, &source_mgr, &context);

  // If errors occur, the library call in the above already logged the error
  // message. So we can just return here.
  if (!module.ok()) return kTrFailure;

  mlir::PassManager pm(&context);

  tensorflow::AddTFToLCETFLConversionPasses(&pm);

  mlir::TFL::QuantizationSpecs quant_specs;

  std::string result;
  auto status = tensorflow::ConvertTFExecutorToTFLOrFlatbuffer(
      module.ValueOrDie().get(), output_mlir, /*emit_builtin_tflite_ops=*/true,
      /*emit_select_tf_ops=*/false, /*emit_custom_ops=*/true, quant_specs,
      &result, &pm);
  if (!status.ok()) return kTrFailure;

  std::string error_msg;
  auto output = mlir::openOutputFile(output_file_name, &error_msg);
  if (output == nullptr) {
    llvm::errs() << error_msg << '\n';
    return kTrFailure;
  }
  output->os() << result;
  output->keep();

  return kTrSuccess;
}
