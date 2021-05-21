#ifndef LARQ_COMPUTE_ENGINE_MLIR_TF_TO_TFL_FLATBUFFER_H_
#define LARQ_COMPUTE_ENGINE_MLIR_TF_TO_TFL_FLATBUFFER_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// This is a fork of ConvertTFExecutorToTFLOrFlatbuffer to enable custom
// OpOrArgLocNameMapper
// https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h#L55-L69
Status ConvertTFExecutorToFlatbuffer(mlir::ModuleOp module, bool export_to_mlir,
                                     std::string* result,
                                     mlir::PassManager* pass_manager);
}  // namespace tensorflow

#endif  // LARQ_COMPUTE_ENGINE_MLIR_TF_TO_TFL_FLATBUFFER_H_
