// RUN: lce-tf-opt %s -tfl-prepare-lce | FileCheck %s --dump-input-on-failure

func @bsign(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sign"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = constant dense<0.1> : tensor<f32>
  %2 = "tf.AddV2"(%0, %cst) : (tensor<8x16xf32>, tensor<f32>) -> tensor<8x16xf32>
  %3 = "tf.Sign"(%2) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %3 : tensor<8x16xf32>
// CHECK-LABEL: bsign
// CHECK:  %0 = "tf.LqceBsign"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK:  return %0
}

func @bconv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst = constant dense<[[[[1.0, -1.0], [1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]]]> : tensor<1x2x2x2xf32>
  %0 = "tf.LqceBsign"(%arg0) : (tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Conv2D"(%0, %cst) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x112x112x2xf32>
  return %1 : tensor<1x112x112x2xf32>
// CHECK-LABEL: bconv2d
// CHECK:  %[[CST1:.*]] = constant dense<-2.000000e+00> : tensor<2xf32>
// CHECK:  %[[CST2:.*]] = constant dense<4.000000e+00> : tensor<2xf32>
// CHECK:  %[[TRP:.*]] = "tf.Transpose"
// CHECK:  %[[CONV:.*]] = "tf.LqceBconv2d64"(%arg0, %[[TRP]], %[[CST1]], %[[CST2]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<2x1x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
// CHECK:  return %[[CONV]]
}

func @notbconv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst = constant dense<0.5> : tensor<1x2x2x2xf32>
  %0 = "tf.LqceBsign"(%arg0) : (tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Conv2D"(%0, %cst) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x112x112x2xf32>
  return %1 : tensor<1x112x112x2xf32>
// CHECK-LABEL: notbconv2d
// CHECK:  %0 = "tf.LqceBsign"(%arg0) : (tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>
// CHECK:  %1 = "tf.Conv2D"
}
