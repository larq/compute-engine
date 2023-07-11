// RUN: lce-tf-opt %s -detection-postprocess-int -verify-diagnostics | FileCheck %s

// CHECK-LABEL: detection_postprocess_int
func.func @detection_postprocess_int(%arg0: tensor<1x10x4x!quant.uniform<i8:f32, 2.343750e-02>>, %arg1: tensor<1x10x1x!quant.uniform<i8:f32, 2.343750e-02>>, %arg2: tensor<10x4x!quant.uniform<i8:f32, 2.343750e-02>>) -> (tensor<1x20x4xi32>, tensor<1x20xi32>, tensor<1x20xf32>, tensor<1xi32>) {
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x10x4x!quant.uniform<i8:f32, 2.343750e-02>>) -> tensor<1x10x4xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x10x1x!quant.uniform<i8:f32, 2.343750e-02>>) -> tensor<1x10x1xf32>
  %2 = "tfl.dequantize"(%arg2) : (tensor<10x4x!quant.uniform<i8:f32, 2.343750e-02>>) -> tensor<10x4xf32>
  %3:4 = "tfl.custom"(%0, %1, %2) {custom_code = "TFLite_Detection_PostProcess", custom_option = #tfl<const_bytes : "0x6D61785F646574656374696F6E73006D61785F636C61737365735F7065725F646574656374696F6E006E756D5F636C6173736573006E6D735F73636F72655F7468726573686F6C64006E6D735F696F755F7468726573686F6C6400795F7363616C6500785F7363616C6500685F7363616C6500775F7363616C65007573655F726567756C61725F6E6D73000A217E8E465B681720313A00000C000000010000000A0000000000803F01000000140000000000003F9A9959BF01000000010000000000803F0000803F0000803F0E06060E0E06060E0E0E322601">} : (tensor<1x10x4xf32>, tensor<1x10x1xf32>, tensor<10x4xf32>) -> (tensor<1x20x4xi32>, tensor<1x20xi32>, tensor<1x20xf32>, tensor<1xi32>)
  return %3#0, %3#1, %3#2, %3#3 : tensor<1x20x4xi32>, tensor<1x20xi32>, tensor<1x20xf32>, tensor<1xi32>  // boxes, classes, scores, num_detections

  // CHECK: %3:4 = "tfl.custom"(%arg0, %arg1, %arg2) {custom_code = "TFLite_Detection_PostProcess", custom_option = #tfl<const_bytes : "0x6D61785F646574656374696F6E73006D61785F636C61737365735F7065725F646574656374696F6E006E756D5F636C6173736573006E6D735F73636F72655F7468726573686F6C64006E6D735F696F755F7468726573686F6C6400795F7363616C6500785F7363616C6500685F7363616C6500775F7363616C65007573655F726567756C61725F6E6D73000A217E8E465B681720313A00000C000000010000000A0000000000803F01000000140000000000003F9A9959BF01000000010000000000803F0000803F0000803F0E06060E0E06060E0E0E322601">} : (tensor<1x10x4x!quant.uniform<i8:f32, 2.343750e-02>>, tensor<1x10x1x!quant.uniform<i8:f32, 2.343750e-02>>, tensor<10x4x!quant.uniform<i8:f32, 2.343750e-02>>) -> (tensor<1x20x4xi32>, tensor<1x20xi32>, tensor<1x20x!quant.uniform<i8:f32, 2.343750e-02>>, tensor<1xi32>)
  // CHECK-NEXT: %4 = "tfl.dequantize"(%3#2) : (tensor<1x20x!quant.uniform<i8:f32, 2.343750e-02>>) -> tensor<1x20xf32>
  // CHECK-NEXT: return %3#0, %3#1, %4, %3#3 : tensor<1x20x4xi32>, tensor<1x20xi32>, tensor<1x20xf32>, tensor<1xi32>
}
