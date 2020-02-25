// RUN: lce-tf-opt %s -tfl-prepare-lce | FileCheck %s --dump-input-on-failure

func @bsign(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sign"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = "tf.Const"() { value = dense<0.1> : tensor<f32> } : () -> tensor<f32>
  %2 = "tf.AddV2"(%0, %cst) : (tensor<8x16xf32>, tensor<f32>) -> tensor<8x16xf32>
  %3 = "tf.Sign"(%2) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %3 : tensor<8x16xf32>
// CHECK-LABEL: bsign
// CHECK:  %0 = "tf.LqceBsign"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
}
