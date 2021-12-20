// RUN: lce-tf-opt %s -tfl-legalize-lce -lce-translate-tfl -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @translate_bconv2d
func @translate_bconv2d(%arg0: tensor<256x32x32x1xi32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: none) -> tensor<256x30x30x16xf32> {
  %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %arg3, %arg4) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %arg3, %arg4) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x1xi32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  // CHECK-NEXT: return %0 : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: @translate_bmax_pool2d
func @translate_bmax_pool2d(%arg0: tensor<256x32x32x3xi32>) -> tensor<256x16x16x3xi32> {
  %0 = "lq.BMaxPool2d"(%arg0) {filter_height = 2 : i32, filter_width = 2 : i32, padding = "SAME", stride_height = 2 : i32, stride_width = 2 : i32} : (tensor<256x32x32x3xi32>) -> tensor<256x16x16x3xi32>
  return %0 : tensor<256x16x16x3xi32>

  // CHECK: %0 = "lq.BMaxPool2d"(%arg0) {filter_height = 2 : i32, filter_width = 2 : i32, padding = "SAME", stride_height = 2 : i32, stride_width = 2 : i32} : (tensor<256x32x32x3xi32>) -> tensor<256x16x16x3xi32>
  // CHECK-NEXT: return %0 : tensor<256x16x16x3xi32>
}

// CHECK-LABEL: @translate_quantize
func @translate_quantize(%arg0: tensor<256x32x32x64xf32>) -> tensor<256x32x32x2xi32> {
  %0 = "lq.Quantize"(%arg0) : (tensor<256x32x32x64xf32>) -> tensor<256x32x32x2xi32>
  return %0 : tensor<256x32x32x2xi32>

  // CHECK: %0 = "lq.Quantize"(%arg0) : (tensor<256x32x32x64xf32>) -> tensor<256x32x32x2xi32>
  // CHECK-NEXT: return %0 : tensor<256x32x32x2xi32>
}

// CHECK-LABEL: @translate_dequantize
func @translate_dequantize(%arg0: tensor<256x32x32x2xi32>) -> tensor<256x32x32x64xf32> {
  %0 = "lq.Dequantize"(%arg0) : (tensor<256x32x32x2xi32>) -> tensor<256x32x32x64xf32>
  return %0 : tensor<256x32x32x64xf32>

  // CHECK: %0 = "lq.Dequantize"(%arg0) : (tensor<256x32x32x2xi32>) -> tensor<256x32x32x64xf32>
  // CHECK-NEXT: return %0 : tensor<256x32x32x64xf32>
}
