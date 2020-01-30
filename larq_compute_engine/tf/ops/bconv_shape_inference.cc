#include "bconv_shape_inference.h"

#include "tensorflow/core/framework/common_shape_fns.h"

Status BConv2DShape(shape_inference::InferenceContext* c, int bitwidth) {
  using namespace shape_inference;

  string filter_format_str;
  if (!c->GetAttr("filter_format", &filter_format_str).ok()) {
    filter_format_str = "HWIO";
  }

  // Normal convolution, handle it using the normal convolution shape inference
  if (filter_format_str == "HWIO") {
    return Conv2DShapeWithExplicitPadding(c);
  }

  bool bitpacked;
  if (filter_format_str == "OHWI") {
    bitpacked = false;
  } else if (filter_format_str == "OHWI_PACKED") {
    bitpacked = true;
  } else {
    return errors::InvalidArgument("Invalid filter format string: ",
                                   filter_format_str);
  }

  // Here we have OHWI or OHWI_PACKED as filter format.
  // This happens when a model is converter by the modelconverter
  // This means we don't have to have a working OP,
  // but we do need the correct shape inference,
  // because it still needs to determine the output shape.

  // The following code is closely based on
  // shape_inference::Conv2DShapeWithExplicitPadding
  // but with the assumption that the data format is
  // NHWC and the filter format is OHWI, optionally bitpacked along channels.
  // This slightly simplifies the code.

  string data_format_str;
  if (!c->GetAttr("data_format", &data_format_str).ok()) {
    data_format_str = "NHWC";
  } else if (data_format_str != "NHWC") {
    return errors::InvalidArgument("Invalid data format string: ",
                                   data_format_str);
  }
  TensorFormat data_format = FORMAT_NHWC;

  constexpr int num_spatial_dims = 2;
  const int rank = GetTensorDimsFromSpatialDims(num_spatial_dims, data_format);
  ShapeHandle conv_input_shape, filter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &conv_input_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), rank, &filter_shape));

  std::vector<int32> dilations;
  TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));

  if (dilations.size() != 4) {
    return errors::InvalidArgument(
        "Conv2D requires the dilation attribute to contain 4 values, but got: ",
        dilations.size());
  }

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

  if (strides.size() != 4) {
    return errors::InvalidArgument("Conv2D on data format ", data_format_str,
                                   " requires the stride attribute to contain"
                                   " 4 values, but got: ",
                                   strides.size());
  }

  const int32 stride_rows = GetTensorDim(strides, data_format, 'H');
  const int32 stride_cols = GetTensorDim(strides, data_format, 'W');
  const int32 dilation_rows = GetTensorDim(dilations, data_format, 'H');
  const int32 dilation_cols = GetTensorDim(dilations, data_format, 'W');

  DimensionHandle batch_size_dim = c->Dim(conv_input_shape, 0);
  DimensionHandle input_rows_dim = c->Dim(conv_input_shape, 1);
  DimensionHandle input_cols_dim = c->Dim(conv_input_shape, 2);
  DimensionHandle input_depth_dim = c->Dim(conv_input_shape, 3);

  DimensionHandle output_depth_dim = c->Dim(filter_shape, 0);
  DimensionHandle filter_rows_dim = c->Dim(filter_shape, 1);
  DimensionHandle filter_cols_dim = c->Dim(filter_shape, 2);
  DimensionHandle filter_input_depth_dim = c->Dim(filter_shape, 3);

  // Check that the input tensor and the filter tensor agree on the channel
  // count.
  if (c->ValueKnown(input_depth_dim) && c->ValueKnown(filter_input_depth_dim)) {
    int64 input_depth_value = c->Value(input_depth_dim),
          filter_input_depth_value = c->Value(filter_input_depth_dim);
    if (bitpacked) {
      // Compute the input depth after bitpacking along depth dimension
      int64 input_depth_value_bp =
          (input_depth_value + bitwidth - 1) / bitwidth;
      if (input_depth_value_bp != filter_input_depth_value) {
        return errors::InvalidArgument(
            "Depth of bitpacked input (", input_depth_value, "->",
            input_depth_value_bp,
            ") is not equal to depth of bitpacked filters (",
            filter_input_depth_value, ")");
      }
    } else {
      if (input_depth_value != filter_input_depth_value) {
        return errors::InvalidArgument(
            "Depth of input (", input_depth_value,
            ") is not equal to depth of bitpacked filters (",
            filter_input_depth_value, ")");
      }
    }
  } else {
    return errors::InvalidArgument(
        "Depth of input or depth of filters is unknown.");
  }

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  std::vector<int64> explicit_paddings;

  Status s = c->GetAttr("explicit_paddings", &explicit_paddings);
  // Use the default value, which is an empty list, if the attribute is not
  // found. Otherwise return the error to the caller.
  if (!s.ok() && !errors::IsNotFound(s)) {
    return s;
  }
  TF_RETURN_IF_ERROR(CheckValidPadding(padding, explicit_paddings,
                                       /*num_dims=*/4, data_format));

  // Compute output shape based on input shape, filter shape, and padding

  DimensionHandle output_rows, output_cols;
  int64 pad_rows_before = -1, pad_rows_after = -1;
  int64 pad_cols_before = -1, pad_cols_after = -1;
  if (padding == Padding::EXPLICIT) {
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H',
                             &pad_rows_before, &pad_rows_after);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W',
                             &pad_cols_before, &pad_cols_after);
  }
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, input_rows_dim, filter_rows_dim, dilation_rows, stride_rows, padding,
      pad_rows_before, pad_rows_after, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, input_cols_dim, filter_cols_dim, dilation_cols, stride_cols, padding,
      pad_cols_before, pad_cols_after, &output_cols));

  ShapeHandle output_shape = c->MakeShape(
      {batch_size_dim, output_rows, output_cols, output_depth_dim});

  c->set_output(0, output_shape);
  return Status::OK();
}
