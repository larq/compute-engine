#include "larq_compute_engine/mlir/tf_tfl_passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_translate_cl.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"

enum TranslationStatus { kTrSuccess, kTrFailure };

int main(int argc, char** argv) {
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

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_file_name, &error_message);
  if (!file) {
    llvm::errs() << error_message << "\n";
    return kTrFailure;
  }

  mlir::MLIRContext context;
  mlir::OwningModuleRef module = tensorflow::GraphdefToMlirTranslateFunction(
      file->getBuffer(), debug_info_file, input_arrays, input_dtypes,
      input_shapes, output_arrays, /*prune_unused_nodes=*/true,
      /*convert_legacy_fed_inputs=*/true, /*graph_as_function=*/false,
      /*upgrade_legacy=*/true, /*add_pseudo_input_nodes=*/false, &context);

  mlir::PassManager pm(&context);
  tensorflow::AddTFToLCETFLConversionPasses(&pm);

  if (failed(pm.run(module.get()))) {
    return kTrFailure;
  }

  std::string result;
  // Write MLIR TFLite dialect into FlatBuffer
  if (tflite::MlirToFlatBufferTranslateFunction(
          module.get(), &result, /*emit_builtin_tflite_ops=*/true,
          /*emit_select_tf_ops=*/false, /*emit_custom_ops=*/true,
          /*add_pseudo_input_nodes=*/false)) {
    return kTrFailure;
  }

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
