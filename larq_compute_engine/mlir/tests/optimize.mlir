// RUN: lce-tf-opt %s -tfl-optimize-lce=target=arm -verify-diagnostics | FileCheck %s --check-prefixes CHECK,CHECK-ARM
// RUN: lce-tf-opt %s -tfl-optimize-lce=target=xcore -verify-diagnostics | FileCheck %s --check-prefixes CHECK,CHECK-XCORE

// CHECK-LABEL: @optimize_quantize_greater_equal_zero
func @optimize_quantize_greater_equal_zero(%arg0: tensor<48x48x64xf32>) -> tensor<48x48x2xi32> {
  %cst = arith.constant dense<0.0> : tensor<f32>
  %0 = "tfl.greater_equal"(%arg0, %cst) : (tensor<48x48x64xf32>, tensor<f32>) -> tensor<48x48x64xi1>
  %1 = "lq.Quantize"(%0) : (tensor<48x48x64xi1>) -> tensor<48x48x2xi32>
  return %1 : tensor<48x48x2xi32>

  // CHECK-NEXT: %0 = "lq.Quantize"(%arg0) : (tensor<48x48x64xf32>) -> tensor<48x48x2xi32>
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @optimize_quantize_greater_equal_non_zero
func @optimize_quantize_greater_equal_non_zero(%arg0: tensor<48x48x64xf32>, %arg1: tensor<48x48x64xf32>) -> tensor<48x48x2xi32> {
  %0 = "tfl.greater_equal"(%arg0, %arg1) : (tensor<48x48x64xf32>, tensor<48x48x64xf32>) -> tensor<48x48x64xi1>
  %1 = "lq.Quantize"(%0) : (tensor<48x48x64xi1>) -> tensor<48x48x2xi32>
  return %1 : tensor<48x48x2xi32>

  // CHECK-NEXT: %0 = tfl.sub %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<48x48x64xf32>
  // CHECK-NEXT: %1 = "lq.Quantize"(%0) : (tensor<48x48x64xf32>) -> tensor<48x48x2xi32>
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @optimize_quantize_less_equal_zero
func @optimize_quantize_less_equal_zero(%arg0: tensor<48x48x64xf32>) -> tensor<48x48x2xi32> {
  %cst = arith.constant dense<0.0> : tensor<64xf32>
  %0 = "tfl.less_equal"(%cst, %arg0) : (tensor<64xf32>, tensor<48x48x64xf32>) -> tensor<48x48x64xi1>
  %1 = "lq.Quantize"(%0) : (tensor<48x48x64xi1>) -> tensor<48x48x2xi32>
  return %1 : tensor<48x48x2xi32>

  // CHECK-NEXT: %0 = "lq.Quantize"(%arg0) : (tensor<48x48x64xf32>) -> tensor<48x48x2xi32>
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @optimize_quantize_less_equal_non_zero
func @optimize_quantize_less_equal_non_zero(%arg0: tensor<48x48x64xf32>, %arg1: tensor<48x48x64xf32>) -> tensor<48x48x2xi32> {
  %0 = "tfl.less_equal"(%arg0, %arg1) : (tensor<48x48x64xf32>, tensor<48x48x64xf32>) -> tensor<48x48x64xi1>
  %1 = "lq.Quantize"(%0) : (tensor<48x48x64xi1>) -> tensor<48x48x2xi32>
  return %1 : tensor<48x48x2xi32>

  // CHECK-NEXT: %0 = tfl.sub %arg1, %arg0 {fused_activation_function = "NONE"} : tensor<48x48x64xf32>
  // CHECK-NEXT: %1 = "lq.Quantize"(%0) : (tensor<48x48x64xf32>) -> tensor<48x48x2xi32>
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @fuse_add_into_bconv2d
func @fuse_add_into_bconv2d(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>, %arg3: none) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<1.5> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %post_activation_bias, %arg3) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = arith.constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]>
  // CHECK-NEXT: %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %cst, %arg3)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_sub_into_bconv2d
func @fuse_sub_into_bconv2d(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>, %arg3: none) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %post_activation_bias, %arg3) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = arith.constant dense<[5.000000e-01, 1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01]>
  // CHECK-NEXT: %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %cst, %arg3)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_div_into_bconv2d
func @fuse_div_into_bconv2d(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<1.5> : tensor<16xf32>
  %post_activation_multiplier = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  %1 = "tfl.div"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = arith.constant dense<[2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01, 2.400000e+01, 2.600000e+01, 2.800000e+01, 3.000000e+01, 3.200000e+01]>
  // CHECK-NEXT: %cst_0 = arith.constant dense<3.000000e+00>
  // CHECK-NEXT: %0 = "lq.Bconv2d"(%arg0, %arg1, %cst, %cst_0, %arg2)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_mul_into_bconv2d
func @fuse_mul_into_bconv2d(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<2.0> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<1.5> : tensor<16xf32>
  %post_activation_multiplier = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = arith.constant dense<[2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01, 2.400000e+01, 2.600000e+01, 2.800000e+01, 3.000000e+01, 3.200000e+01]>
  // CHECK-NEXT: %cst_0 = arith.constant dense<3.000000e+00>
  // CHECK-NEXT: %0 = "lq.Bconv2d"(%arg0, %arg1, %cst, %cst_0, %arg2)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_relu_into_bconv2d
func @fuse_relu_into_bconv2d(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = arith.constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<0.0> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %arg1, %cst, %cst_0, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "RELU", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32}
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_relu6_into_bconv2d
func @fuse_relu6_into_bconv2d(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = arith.constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<0.0> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu6"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %arg1, %cst, %cst_0, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "RELU6", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32}
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_relu1_into_bconv2d
func @fuse_relu1_into_bconv2d(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = arith.constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<0.0> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu_n1_to_1"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %arg1, %cst, %cst_0, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "RELU_N1_TO_1", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32}
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_relu_into_bconv2d_padding_same_one
func @fuse_relu_into_bconv2d_padding_same_one(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x32x32x16xf32> {
  %post_activation_multiplier = arith.constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<0.0> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 1 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x32x32x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x32x32x16xf32>) -> tensor<256x32x32x16xf32>
  return %1 : tensor<256x32x32x16xf32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %arg1, %cst, %cst_0, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "RELU", pad_values = 1 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32}
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @do_not_fuse_relu_into_bconv2d_padding_same_zero
func @do_not_fuse_relu_into_bconv2d_padding_same_zero(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x32x32x16xf32> {
  %post_activation_multiplier = arith.constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<0.0> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x32x32x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x32x32x16xf32>) -> tensor<256x32x32x16xf32>
  return %1 : tensor<256x32x32x16xf32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %arg1, %cst, %cst_0, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32}
  // CHECK-NEXT: %1 = "tfl.relu"(%0)
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @do_not_fuse_relu_into_bconv2d_no_post_activation_bias
func @do_not_fuse_relu_into_bconv2d_no_post_activation_bias(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = arith.constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<5.0> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %arg1, %cst, %cst_0, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32}
  // CHECK-NEXT: %1 = "tfl.relu"(%0)
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @do_not_fuse_relu_into_bconv2d_no_post_activation_multiplier
func @do_not_fuse_relu_into_bconv2d_no_post_activation_multiplier(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = arith.constant dense<0.8> : tensor<16xf32>
  %post_activation_bias = arith.constant dense<0.0> : tensor<16xf32>
  %0 = "lq.Bconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %arg1, %cst, %cst_0, %arg2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32}
  // CHECK-NEXT: %1 = "tfl.relu"(%0)
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @target_specific_reorder_maxpool_2d_quantize
func @target_specific_reorder_maxpool_2d_quantize(%arg0: tensor<256x32x32x65xf32>) -> tensor<256x16x8x3xi32> {
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 2 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 4 : i32} : (tensor<256x32x32x65xf32>) -> tensor<256x16x8x65xf32>
  %1 = "lq.Quantize"(%0) : (tensor<256x16x8x65xf32>) -> tensor<256x16x8x3xi32>
  return %1 : tensor<256x16x8x3xi32>

  // CHECK-ARM-NEXT: %0 = "lq.Quantize"(%arg0) : (tensor<256x32x32x65xf32>) -> tensor<256x32x32x3xi32>
  // CHECK-ARM-NEXT: %1 = "lq.BMaxPool2d"(%0) {filter_height = 3 : i32, filter_width = 2 : i32, padding = "SAME", stride_height = 2 : i32, stride_width = 4 : i32} : (tensor<256x32x32x3xi32>) -> tensor<256x16x8x3xi32>
  // CHECK-ARM-NEXT: return %1

  // CHECK-XCORE-NEXT: %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 2 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 4 : i32} : (tensor<256x32x32x65xf32>) -> tensor<256x16x8x65xf32>
  // CHECK-XCORE-NEXT: %1 = "lq.Quantize"(%0) : (tensor<256x16x8x65xf32>) -> tensor<256x16x8x3xi32>
  // CHECK-XCORE-NEXT: return %1
}

// CHECK-LABEL: @do_not_reorder_maxpool_2d_quantize_multiple_uses
func @do_not_reorder_maxpool_2d_quantize_multiple_uses(%arg0: tensor<256x32x32x65xf32>) -> (tensor<256x16x8x65xf32>, tensor<256x16x8x3xi32>) {
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 2 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 4 : i32} : (tensor<256x32x32x65xf32>) -> tensor<256x16x8x65xf32>
  %1 = "lq.Quantize"(%0) : (tensor<256x16x8x65xf32>) -> tensor<256x16x8x3xi32>
  return %0, %1 : tensor<256x16x8x65xf32>, tensor<256x16x8x3xi32>

  // CHECK-NEXT: %0 = "tfl.max_pool_2d"(%arg0)
  // CHECK-NEXT: %1 = "lq.Quantize"(%0)
  // CHECK-NEXT: return %0, %1
}

// CHECK-LABEL: @bitpack_activation_thresholds_with_negative_post_multipliers
func @bitpack_activation_thresholds_with_negative_post_multipliers(%arg0: tensor<256x32x32x1xi32>) -> tensor<256x32x32x1xi32> {
  %filter = arith.constant dense<1.0> : tensor<8x2x2x1xf32>
  %post_activation_multiplier = arith.constant dense<[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]> : tensor<8xf32>
  %post_activation_bias = arith.constant dense<[-10.0, 8.0, 0.4, 1.0, -0.01, 0.5, -1.0, 2.71]> : tensor<8xf32>
  %cst = constant unit
  %0 = "lq.Bconv2d"(%arg0, %filter, %post_activation_multiplier, %post_activation_bias, %cst) {channels_in = 1 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 1 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<8x2x2x1xf32>, tensor<8xf32>, tensor<8xf32>, none) -> tensor<256x32x32x8xf32>
  %1 = "lq.Quantize"(%0) : (tensor<256x32x32x8xf32>) -> tensor<256x32x32x1xi32>
  return %1 : tensor<256x32x32x1xi32>

  // The new filter values. Check that the values for the first four filters
  // have had their signs flipped. The syntax is very ugly because filecheck
  // tries to perform string substitution when it encounters `[[` and so we have
  // to wrap the whole thing in a regex, and escape the square brackets.
  // CHECK: %cst = constant {{dense<\[\[\[\[-1.000000e\+00\], \[-1.000000e\+00\]\], \[\[-1.000000e\+00\], \[-1.000000e\+00\]\]\], \[\[\[-1.000000e\+00\], \[-1.000000e\+00\]\], \[\[-1.000000e\+00\], \[-1.000000e\+00\]\]\], \[\[\[-1.000000e\+00\], \[-1.000000e\+00\]\], \[\[-1.000000e\+00\], \[-1.000000e\+00\]\]\], \[\[\[-1.000000e\+00\], \[-1.000000e\+00\]\], \[\[-1.000000e\+00\], \[-1.000000e\+00\]\]\], \[\[\[1.000000e\+00\], \[1.000000e\+00\]\], \[\[1.000000e\+00\], \[1.000000e\+00\]\]\], \[\[\[1.000000e\+00\], \[1.000000e\+00\]\], \[\[1.000000e\+00\], \[1.000000e\+00\]\]\], \[\[\[1.000000e\+00\], \[1.000000e\+00\]\], \[\[1.000000e\+00\], \[1.000000e\+00\]\]\], \[\[\[1.000000e\+00\], \[1.000000e\+00\]\], \[\[1.000000e\+00\], \[1.000000e\+00\]\]\]\]>}} : tensor<8x2x2x1xf32>

  // The `none` value that will replace the post-multiplier and post-bias.
  // CHECK-NEXT: %cst_0 = constant unit

  // Verify correct thresholds. These have been manually computed.
  // CHECK-NEXT: %cst_1 = arith.constant dense<[0, 3, 2, 2, -2147483648, 2, 1, 2]> : tensor<8xi32>

  // CHECK-NEXT: %0 = "lq.Bconv2d"(%arg0, %cst, %cst_0, %cst_0, %cst_1) {channels_in = 1 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 1 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<8x2x2x1xf32>, none, none, tensor<8xi32>) -> tensor<256x32x32x1xi32>
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @bitpack_activations_valid_padding
func @bitpack_activations_valid_padding(%arg0: tensor<256x32x32x1xi32>) -> tensor<256x30x30x3xi32> {
  %filter = arith.constant dense<1.0> : tensor<65x3x3x3xf32>
  %post_activation_multiplier = arith.constant dense<0.5> : tensor<65xf32>
  %post_activation_bias = arith.constant dense<-1.0> : tensor<65xf32>
  %cst = constant unit
  %0 = "lq.Bconv2d"(%arg0, %filter, %post_activation_multiplier, %post_activation_bias, %cst) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>, none) -> tensor<256x30x30x65xf32>
  %1 = "lq.Quantize"(%0) : (tensor<256x30x30x65xf32>) -> tensor<256x30x30x3xi32>
  return %1 : tensor<256x30x30x3xi32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %cst, %cst_0, %cst_0, %cst_1) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<65x3x3x3xf32>, none, none, tensor<65xi32>) -> tensor<256x30x30x3xi32>
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @bitpack_activations_same_one_padding
func @bitpack_activations_same_one_padding(%arg0: tensor<256x32x32x1xi32>) -> tensor<256x32x32x3xi32> {
  %filter = arith.constant dense<1.0> : tensor<65x3x3x3xf32>
  %post_activation_multiplier = arith.constant dense<0.5> : tensor<65xf32>
  %post_activation_bias = arith.constant dense<-1.0> : tensor<65xf32>
  %cst = constant unit
  %0 = "lq.Bconv2d"(%arg0, %filter, %post_activation_multiplier, %post_activation_bias, %cst) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 1 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>, none) -> tensor<256x32x32x65xf32>
  %1 = "lq.Quantize"(%0) : (tensor<256x32x32x65xf32>) -> tensor<256x32x32x3xi32>
  return %1 : tensor<256x32x32x3xi32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %cst, %cst_0, %cst_0, %cst_1) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 1 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<65x3x3x3xf32>, none, none, tensor<65xi32>) -> tensor<256x32x32x3xi32>
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @do_not_bitpack_activations_same_zero_padding
func @do_not_bitpack_activations_same_zero_padding(%arg0: tensor<256x32x32x1xi32>) -> tensor<256x32x32x3xi32> {
  %filter = arith.constant dense<1.0> : tensor<65x3x3x3xf32>
  %post_activation_multiplier = arith.constant dense<0.5> : tensor<65xf32>
  %post_activation_bias = arith.constant dense<-1.0> : tensor<65xf32>
  %cst = constant unit
  %0 = "lq.Bconv2d"(%arg0, %filter, %post_activation_multiplier, %post_activation_bias, %cst) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>, none) -> tensor<256x32x32x65xf32>
  %1 = "lq.Quantize"(%0) : (tensor<256x32x32x65xf32>) -> tensor<256x32x32x3xi32>
  return %1 : tensor<256x32x32x3xi32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %cst, %cst_0, %cst_1, %cst_2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>, none) -> tensor<256x32x32x65xf32>
  // CHECK-NEXT: %1 = "lq.Quantize"(%0) : (tensor<256x32x32x65xf32>) -> tensor<256x32x32x3xi32>
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @do_not_bitpack_activations_multiple_uses
func @do_not_bitpack_activations_multiple_uses(%arg0: tensor<256x32x32x1xi32>) -> (tensor<256x30x30x65xf32>, tensor<256x30x30x3xi32>) {
  %filter = arith.constant dense<1.0> : tensor<65x3x3x3xf32>
  %post_activation_multiplier = arith.constant dense<0.5> : tensor<65xf32>
  %post_activation_bias = arith.constant dense<-1.0> : tensor<65xf32>
  %cst = constant unit
  %0 = "lq.Bconv2d"(%arg0, %filter, %post_activation_multiplier, %post_activation_bias, %cst) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>, none) -> tensor<256x30x30x65xf32>
  %1 = "lq.Quantize"(%0) : (tensor<256x30x30x65xf32>) -> tensor<256x30x30x3xi32>
  return %0, %1: tensor<256x30x30x65xf32>, tensor<256x30x30x3xi32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %cst, %cst_0, %cst_1, %cst_2) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>, none) -> tensor<256x30x30x65xf32>
  // CHECK-NEXT: %1 = "lq.Quantize"(%0) : (tensor<256x30x30x65xf32>) -> tensor<256x30x30x3xi32>
  // CHECK-NEXT: return %0, %1
}
