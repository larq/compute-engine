#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/dilated_conv.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

using compute_engine::core::bitpacking_bitwidth;

// Prepare LCE operations in functions for subsequent legalization.
struct PrepareLCE : public PassWrapper<PrepareLCE, FunctionPass> {
  void runOnFunction() override;
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::lq::LarqDialect>();
  }
};

DenseElementsAttr GetConstantVector(Attribute filter, float val) {
  auto filter_type = filter.getType().cast<ShapedType>();
  auto filter_shape = filter_type.getShape();

  RankedTensorType type =
      RankedTensorType::get({filter_shape[3]}, filter_type.getElementType());
  return DenseElementsAttr::get(type, val);
}

DenseElementsAttr GetScaleVector(Attribute filter_attr) {
  auto filter = filter_attr.cast<DenseElementsAttr>();
  auto filter_type = filter_attr.getType().cast<ShapedType>();
  auto channels = filter_type.getShape()[3];
  auto element_type = filter_type.getElementType();

  std::vector<Attribute> scales(channels);
  for (std::size_t i = 0; i < channels; ++i) {
    auto scale = std::abs(filter.getValue<float>({0, 0, 0, i}));
    scales[i] = FloatAttr::get(element_type, scale);
  }

  RankedTensorType type = RankedTensorType::get({channels}, element_type);
  return DenseElementsAttr::get(type, scales);
}

bool IsBinaryFilter(Attribute filter_attr) {
  if (!filter_attr.isa<DenseElementsAttr>()) return false;
  auto filter = filter_attr.cast<DenseElementsAttr>();

  auto shape = filter_attr.getType().cast<ShapedType>().getShape();
  if (shape.size() != 4) return false;

  for (std::size_t h = 0; h < shape[0]; ++h) {
    for (std::size_t w = 0; w < shape[1]; ++w) {
      for (std::size_t i = 0; i < shape[2]; ++i) {
        for (std::size_t o = 0; o < shape[3]; ++o) {
          auto scale = filter.getValue<float>({0, 0, 0, o});
          if (std::abs(scale) <= std::numeric_limits<float>::epsilon())
            return false;
          auto value = filter.getValue<float>({h, w, i, o});
          if (std::abs(std::abs(value / scale) - 1.0f) > 0.005f) return false;
        }
      }
    }
  }
  return true;
}

bool IsSamePadding(Attribute paddings_attr, Value input, Value output,
                   ArrayAttr strides_attr) {
  if (!paddings_attr.isa<DenseElementsAttr>()) return false;
  auto paddings = paddings_attr.dyn_cast<DenseElementsAttr>();
  auto input_shape = input.getType().cast<RankedTensorType>().getShape();
  auto output_shape = output.getType().cast<RankedTensorType>().getShape();
  auto strides = strides_attr.getValue();

  int pad_height_left = paddings.getValue<int>({1, 0});
  int pad_height_right = paddings.getValue<int>({1, 1});
  int pad_width_left = paddings.getValue<int>({2, 0});
  int pad_width_right = paddings.getValue<int>({2, 1});

  int pad_height = pad_height_left + pad_height_right;
  int pad_width = pad_width_left + pad_width_right;

  int stride_height = strides[1].cast<IntegerAttr>().getInt();
  int stride_width = strides[2].cast<IntegerAttr>().getInt();

  return paddings.getValue<int>({0, 0}) == 0 &&
         paddings.getValue<int>({0, 1}) == 0 &&
         output_shape[1] ==
             (input_shape[1] + stride_height - 1) / stride_height &&
         output_shape[2] ==
             (input_shape[2] + stride_width - 1) / stride_width &&
         pad_height_left == pad_height / 2 &&
         pad_height_right == (pad_height + 1) / 2 &&
         pad_width_left == pad_width / 2 &&
         pad_width_right == (pad_width + 1) / 2 &&
         paddings.getValue<int>({3, 0}) == 0 &&
         paddings.getValue<int>({3, 1}) == 0;
}

// Verify that the filter shape is compatible with the input shape. Will fail if
// any other type is passed. Will emit an error and return false if the two
// shapes are incompatible (specifically, if the shapes imply a grouped
// convolution with a group-shape that is not a multiple of 32).
bool HasValidFilterShape(Value input_val, Value filter_val) {
  auto input_type = input_val.getType().cast<ShapedType>();
  auto input_shape_vector = input_type.getShape();
  auto total_input_channels = input_shape_vector[input_shape_vector.size() - 1];
  auto filter_type = filter_val.getType().cast<ShapedType>();
  auto filter_shape_vector = filter_type.getShape();
  auto filter_input_channels =
      filter_shape_vector[filter_shape_vector.size() - 2];
  if (total_input_channels % filter_input_channels != 0) {
    mlir::emitError(filter_val.getLoc())
        << "Filter dimensions invalid: the number of filter input channels "
        << filter_input_channels
        << " does not divide the total number of input channels "
        << total_input_channels << "\n";
    return false;
  }
  auto num_groups = total_input_channels / filter_input_channels;
  if (num_groups > 1 && filter_input_channels % bitpacking_bitwidth != 0) {
    mlir::emitError(filter_val.getLoc())
        << "Invalid binary grouped convolution: the number of input channels "
           "per-group must be a multiple of "
        << bitpacking_bitwidth << ", but is " << filter_input_channels << "\n";
    return false;
  }
  return true;
}

// Returns the number of channels of a shaped tensor. Will fail if any other
// type is passed.
IntegerAttr GetNumChannels(Builder& b, Value output_val) {
  auto output_type = output_val.getType().cast<ShapedType>();
  auto shape_vector = output_type.getShape();
  return b.getI32IntegerAttr(shape_vector[shape_vector.size() - 1]);
}

#include "larq_compute_engine/mlir/transforms/generated_prepare.inc"

void PrepareLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  // This pattern will try to identify and optimize for dilated convolution.
  // e.g. Patterns like "SpaceToBatchND -> Conv2D -> BatchToSpaceND" will be
  // replaced with a single Conv op with dilation parameter.
  patterns.insert<ConvertTFDilatedConvOp<TF::Conv2DOp>>(ctx);

  TFL::populateWithGenerated(ctx, patterns);
  applyPatternsAndFoldGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect PrepareLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareLCEPass() {
  return std::make_unique<PrepareLCE>();
}

static PassRegistration<PrepareLCE> pass("tfl-prepare-lce", "Inject LCE Ops");

}  // namespace TFL
}  // namespace mlir
