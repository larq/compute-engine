#include "larq_compute_engine/cc/core/bgemm_functor.h"
#include "larq_compute_engine/cc/core/bconv2d_functor.h"
#include "larq_compute_engine/cc/core/padding_functor.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;

namespace ce = compute_engine;

namespace compute_engine {
namespace kernels {

template <class T, class TConvFunctor>
class BConv2DOp : public BinaryOp<T> {
 public:
  explicit BConv2DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Data format not supported by this kernel", data_format));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(
          context,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(context, in_depth == filter.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", filter.dim_size(2)));

    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(1));

    // The first dimension for input is batch.
    const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "Conv2D: in_depth = " << in_depth
            << ", input_cols = " << input_cols
            << ", filter_cols = " << filter_cols
            << ", input_rows = " << input_rows
            << ", filter_rows = " << filter_rows
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", out_depth = " << out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    const T* input_data = input.flat<T>().data();
    const T* filter_data = filter.flat<T>().data();
    T* output_data = output->flat<T>().data();

    TConvFunctor conv_functor;
    conv_functor(input_data, batch, input_rows, input_cols, in_depth,
                 filter_data, filter_rows, filter_cols, out_depth, stride_rows,
                 stride_cols, padding_, output_data, out_rows, out_cols);

    if (padding_ != 1) {
      ce::core::ReferencePaddingFunctor<T, T, ce::core::FilterFormat::HWIO>
          padding_functor;
      padding_functor(batch, input_rows, input_cols, in_depth, filter_data,
                      filter_rows, filter_cols, out_depth, stride_rows,
                      stride_cols, output_data, out_rows, out_cols);
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(BConv2DOp);
};

}  // namespace kernels
}  // namespace compute_engine

#define REGISTER_BITPACKED_KERNEL_CPU(T)                               \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("LqceBconv2d8").Device(DEVICE_CPU).TypeConstraint<T>("T"),  \
      ce::kernels::BConv2DOp<                                          \
          T, ce::core::Im2ColBConvFunctor<                             \
                 T, T, T,                                              \
                 ce::core::FusedBGemmFunctor<                          \
                     T, ce::core::Layout::RowMajor, T,                 \
                     ce::core::Layout::RowMajor, T, std::uint8_t,      \
                     ce::core::ReferenceBGemmFunctor>>>);              \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("LqceBconv2d32").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ce::kernels::BConv2DOp<                                          \
          T, ce::core::Im2ColBConvFunctor<                             \
                 T, T, T,                                              \
                 ce::core::FusedBGemmFunctor<                          \
                     T, ce::core::Layout::RowMajor, T,                 \
                     ce::core::Layout::RowMajor, T, std::uint32_t,     \
                     ce::core::ReferenceBGemmFunctor>>>);              \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("LqceBconv2d64").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ce::kernels::BConv2DOp<                                          \
          T, ce::core::Im2ColBConvFunctor<                             \
                 T, T, T,                                              \
                 ce::core::FusedBGemmFunctor<                          \
                     T, ce::core::Layout::RowMajor, T,                 \
                     ce::core::Layout::RowMajor, T, std::uint64_t,     \
                     ce::core::ReferenceBGemmFunctor>>>);

TF_CALL_float(REGISTER_BITPACKED_KERNEL_CPU);
TF_CALL_double(REGISTER_BITPACKED_KERNEL_CPU);
