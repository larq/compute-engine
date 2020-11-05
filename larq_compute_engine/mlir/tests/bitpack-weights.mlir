// RUN: lce-tf-opt %s -tfl-lce-bitpack-weights -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @bitpack_bconv2d_filters
func @bitpack_bconv2d_filters(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: none) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<1.0> : tensor<16x3x3x3xf32>
  %0 = "lq.Bconv2d"(%arg0, %cst, %arg1, %arg2, %arg3) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<0> : tensor<16x3x3x1xi32>
  // CHECK: %0 = "lq.Bconv2d"(%arg0, %cst, %arg1, %arg2, %arg3) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x1xi32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  // CHECK-NEXT: return %0
}
