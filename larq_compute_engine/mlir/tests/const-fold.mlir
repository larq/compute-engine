// RUN: lce-tf-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @quantize
func.func @quantize() -> (tensor<1x1x2x1xi32>, tensor<1x1x2x1xi32>) {
  %pos = arith.constant dense< 0.5> : tensor<1x1x2x32xf32>
  %neg = arith.constant dense<-0.5> : tensor<1x1x2x32xf32>
  %0 = "lq.Quantize"(%pos) {} : (tensor<1x1x2x32xf32>) -> tensor<1x1x2x1xi32>
  %1 = "lq.Quantize"(%neg) {} : (tensor<1x1x2x32xf32>) -> tensor<1x1x2x1xi32>
  return %0, %1 : tensor<1x1x2x1xi32>, tensor<1x1x2x1xi32>

  // CHECK: %[[neg:.*]] = arith.constant dense<-1> : tensor<1x1x2x1xi32>
  // CHECK: %[[pos:.*]] = arith.constant dense<0> : tensor<1x1x2x1xi32>
  // CHECK: return %[[pos]], %[[neg]] : tensor<1x1x2x1xi32>, tensor<1x1x2x1xi32>
}

// CHECK-LABEL: @dequantize
func.func @dequantize() -> (tensor<1x1x2x32xf32>, tensor<1x1x2x32xf32>) {
  %pos = arith.constant dense<0> : tensor<1x1x2x1xi32>
  %neg = arith.constant dense<-1> : tensor<1x1x2x1xi32>
  %0 = "lq.Dequantize"(%pos) {} : (tensor<1x1x2x1xi32>) -> tensor<1x1x2x32xf32>
  %1 = "lq.Dequantize"(%neg) {} : (tensor<1x1x2x1xi32>) -> tensor<1x1x2x32xf32>
  return %0, %1 : tensor<1x1x2x32xf32>, tensor<1x1x2x32xf32>

  // CHECK: %[[neg:.*]] = arith.constant dense<-1.000000e+00> : tensor<1x1x2x32xf32>
  // CHECK: %[[pos:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1x2x32xf32>
  // CHECK: return %[[pos]], %[[neg]] : tensor<1x1x2x32xf32>, tensor<1x1x2x32xf32>
}
