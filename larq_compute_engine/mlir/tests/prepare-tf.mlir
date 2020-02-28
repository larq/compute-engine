// RUN: lce-tf-opt %s -tfl-prepare-lce | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: @fuse_bsign
func @fuse_bsign(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sign"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = constant dense<0.1> : tensor<f32>
  %2 = "tf.AddV2"(%0, %cst) : (tensor<8x16xf32>, tensor<f32>) -> tensor<8x16xf32>
  %3 = "tf.Sign"(%2) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %3 : tensor<8x16xf32>

  // CHECK-NEXT: %0 = "tf.LqceBsign"(%arg0)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_bconv2d
func @fuse_bconv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst = constant dense<[[[[1.0, -1.0], [1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]]]> : tensor<1x2x2x2xf32>
  %0 = "tf.LqceBsign"(%arg0) : (tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Conv2D"(%0, %cst) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x112x112x2xf32>
  return %1 : tensor<1x112x112x2xf32>

  // CHECK: %cst = constant
  // CHECK: %[[fused_multiply:.*]] = constant dense<-2.000000e+00> : tensor<2xf32>
  // CHECK: %[[fused_add:.*]] = constant dense<4.000000e+00> : tensor<2xf32>
  // CHECK-NEXT: %[[transpose:.*]] = "tf.Transpose"(%cst
  // CHECK-NEXT: %[[conv:.*]] = "tf.LqceBconv2d64"(%arg0, %[[transpose]], %[[fused_multiply]], %[[fused_add]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<2x1x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  // CHECK-NEXT: return %[[conv]]
}

// CHECK-LABEL: @do_not_fuse_bconv2d
func @do_not_fuse_bconv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst = constant dense<0.5> : tensor<1x2x2x2xf32>
  %0 = "tf.LqceBsign"(%arg0) : (tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Conv2D"(%0, %cst) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x112x112x2xf32>
  return %1 : tensor<1x112x112x2xf32>

  // CHECK-NEXT: %cst = constant
  // CHECK-NEXT: %0 = "tf.LqceBsign"(%arg0)
  // CHECK-NEXT: %1 = "tf.Conv2D"(%0, %cst)
  // CHECK-NEXT: return %1
}
