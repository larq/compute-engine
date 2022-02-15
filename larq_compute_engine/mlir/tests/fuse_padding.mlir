// RUN: lce-tf-opt %s -tfl-fuse-padding -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @fuse_pad_into_conv_valid
func @fuse_pad_into_conv_valid(%arg0: tensor<1x64x64x8xf32>) -> tensor<1x64x64x16xf32> {
  %cst0 = constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %cst1 = constant dense<1.0> : tensor<16x3x3x8xf32>
  %cst2 = constant dense<1.0> : tensor<16xf32>
  %0 = "tfl.pad"(%arg0, %cst0) : (tensor<1x64x64x8xf32>, tensor<4x2xi32>) -> tensor<1x66x66x8xf32>
  %1 = "tfl.conv_2d"(%0, %cst1, %cst2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x66x66x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x64x64x16xf32>
  return %1 : tensor<1x64x64x16xf32>

  // CHECK: %0 = "tfl.conv_2d"(%arg0, %cst, %cst_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x64x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x64x64x16xf32>
  // CHECK-NEXT: return %0 : tensor<1x64x64x16xf32>
}

// CHECK-LABEL: @fuse_padv2_into_conv_valid
func @fuse_padv2_into_conv_valid(%arg0: tensor<1x64x64x8xf32>) -> tensor<1x64x64x16xf32> {
  %cst0 = constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %cst1 = constant dense<0.0> : tensor<f32>
  %cst2 = constant dense<1.0> : tensor<16x3x3x8xf32>
  %cst3 = constant dense<1.0> : tensor<16xf32>
  %0 = "tfl.padv2"(%arg0, %cst0, %cst1) : (tensor<1x64x64x8xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<1x66x66x8xf32>
  %1 = "tfl.conv_2d"(%0, %cst2, %cst3) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x66x66x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x64x64x16xf32>
  return %1 : tensor<1x64x64x16xf32>

  // CHECK: %0 = "tfl.conv_2d"(%arg0, %cst, %cst_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x64x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x64x64x16xf32>
  // CHECK-NEXT: return %0 : tensor<1x64x64x16xf32>
}

// CHECK-LABEL: @do_not_fuse_padv2_into_conv_wrong_pad_value
func @do_not_fuse_padv2_into_conv_wrong_pad_value(%arg0: tensor<1x64x64x8xf32>) -> tensor<1x64x64x16xf32> {
  %cst0 = constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %cst1 = constant dense<1.0> : tensor<f32>
  %cst2 = constant dense<1.0> : tensor<16x3x3x8xf32>
  %cst3 = constant dense<1.0> : tensor<16xf32>
  %0 = "tfl.padv2"(%arg0, %cst0, %cst1) : (tensor<1x64x64x8xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<1x66x66x8xf32>
  %1 = "tfl.conv_2d"(%0, %cst2, %cst3) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x66x66x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x64x64x16xf32>
  return %1 : tensor<1x64x64x16xf32>

  // CHECK: %0 = "tfl.padv2"(%arg0, %cst, %cst_0)
}

// CHECK-LABEL: @do_not_fuse_pad_into_conv_same
func @do_not_fuse_pad_into_conv_same(%arg0: tensor<1x64x64x8xf32>) -> tensor<1x66x66x16xf32> {
  %cst0 = constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %cst1 = constant dense<1.0> : tensor<f32>
  %cst2 = constant dense<1.0> : tensor<16x3x3x8xf32>
  %cst3 = constant dense<1.0> : tensor<16xf32>
  %0 = "tfl.padv2"(%arg0, %cst0, %cst1) : (tensor<1x64x64x8xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<1x66x66x8xf32>
  %1 = "tfl.conv_2d"(%0, %cst2, %cst3) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x66x66x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x66x66x16xf32>
  return %1 : tensor<1x66x66x16xf32>

  // CHECK: %0 = "tfl.padv2"(%arg0, %cst, %cst_0)
}
