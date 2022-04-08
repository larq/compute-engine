// RUN: lce-tf-opt %s -lce-quantize -verify-diagnostics | FileCheck %s

// CHECK-LABEL: quantize_bconv2d
func.func @quantize_bconv2d(%arg0: tensor<1x224x224x1xi32>, %arg1: tensor<32x3x3x1xi32>, %arg2: none) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
  %cst0 = arith.constant dense<-1.23697901> : tensor<32xf32>
  %0 = "tfl.quantize"(%cst0) {qtype = tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<32xf32>) -> tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<32xf32>
  %cst1 = arith.constant dense<1.10976315> : tensor<32xf32>
  %2 = "tfl.quantize"(%cst1) {qtype = tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<32xf32>) -> tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %3 = "tfl.dequantize"(%2) : (tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<32xf32>
  %4 = "lq.Bconv2d"(%arg0, %arg1, %1, %3, %arg2) {channels_in = 3 : i32, dilation_height_factor = 2 : i32, dilation_width_factor = 3 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 4 : i32, stride_width = 5 : i32} : (tensor<1x224x224x1xi32>, tensor<32x3x3x1xi32>, tensor<32xf32>, tensor<32xf32>, none) -> tensor<1x112x112x32xf32>
  %5 = "tfl.quantize"(%4) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %5 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst0:.*]] = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %[[cst1:.*]] = arith.constant dense<1.10976315> : tensor<32xf32>
// CHECK: %[[conv:.*]] = "lq.Bconv2d"(%arg0, %arg1, %[[cst0]], %[[cst1]], %arg2)
// CHECK: return %[[conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// CHECK-LABEL: quantize_bitpacked_bconv2d
func.func @quantize_bitpacked_bconv2d(%arg0: tensor<1x224x224x1xi32>, %arg1: tensor<32x3x3x1xi32>, %arg2: none, %arg3: none, %arg4: tensor<32xi32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
  %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %arg3, %arg4) {channels_in = 3 : i32, dilation_height_factor = 2 : i32, dilation_width_factor = 3 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 4 : i32, stride_width = 5 : i32} : (tensor<1x224x224x1xi32>, tensor<32x3x3x1xi32>, none, none, tensor<32xi32>) -> tensor<1x112x112x32xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %1 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK-NEXT: %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %arg3, %arg4)
// CHECK-NEXT: return %0 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// CHECK-LABEL: quantize_lce_dequantize
func.func @quantize_lce_dequantize(%arg0: tensor<1x112x112x1xi32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
  %0 = "lq.Dequantize"(%arg0) : (tensor<1x112x112x1xi32>) -> tensor<1x112x112x32xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %1 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK-NEXT: %0 = "lq.Dequantize"(%arg0) : (tensor<1x112x112x1xi32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
// CHECK-NEXT: return %0 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// CHECK-LABEL: dequantize_lce_quantize
func.func @dequantize_lce_quantize(%arg0: tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x112x112x1xi32> {
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x112x112x32xf32>
  %1 = "lq.Quantize"(%0) : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x1xi32>
  return %1 : tensor<1x112x112x1xi32>

// CHECK: %[[quant:.*]] = "lq.Quantize"(%arg0) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x112x112x1xi32>
// CHECK-NEXT: return %[[quant]] : tensor<1x112x112x1xi32>
}
