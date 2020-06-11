// RUN: lce-tf-opt %s -lce-quantize | FileCheck %s

// CHECK-LABEL: quantize_bconv2d
func @quantize_bconv2d(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<32x3x3x1xi32>, none) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>, %arg1: tensor<32x3x3x1xi32>, %arg2: none):
  %1 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %cst0 = constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.quantize"(%cst0) {qtype = tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<32xf32>) -> tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %3 = "tfl.dequantize"(%2) : (tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<32xf32>
  %cst1 = constant dense<1.10976315> : tensor<32xf32>
  %4 = "tfl.quantize"(%cst1) {qtype = tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<32xf32>) -> tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %5 = "tfl.dequantize"(%4) : (tensor<32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<32xf32>
  %6 = "lq.Bconv2d"(%1, %arg1, %3, %5, %arg2) {channels_in = 3 : i32, dilation_height_factor = 2 : i32, dilation_width_factor = 3 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 4 : i32, stride_width = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x1xi32>, tensor<32xf32>, tensor<32xf32>, none) -> tensor<1x112x112x32xf32>
  %7 = "tfl.quantize"(%6) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %7 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst0:.*]] = constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %[[cst1:.*]] = constant dense<1.10976315> : tensor<32xf32>
// CHECK: %[[conv:.*]] = "lq.Bconv2d"(%arg0, %arg1, %[[cst0]], %[[cst1]], %arg2)
// CHECK: return %[[conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// CHECK-LABEL: quantize_bitpacked_bconv2d
func @quantize_bitpacked_bconv2d(%arg0: tensor<1x224x224x1xi32>, %arg1: tensor<32x3x3x1xi32>, %arg2: none, %arg3: none, %arg4: tensor<32xi32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
  %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %arg3, %arg4) {channels_in = 3 : i32, dilation_height_factor = 2 : i32, dilation_width_factor = 3 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 4 : i32, stride_width = 5 : i32} : (tensor<1x224x224x1xi32>, tensor<32x3x3x1xi32>, none, none, tensor<32xi32>) -> tensor<1x112x112x32xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %1 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK-NEXT: %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %arg3, %arg4)
// CHECK-NEXT: return %0 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}
