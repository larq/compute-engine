// RUN: lce-tf-opt %s -tfl-prepare-lce | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: @fuse_bsign
func @fuse_bsign(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sign"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = constant dense<0.1> : tensor<f32>
  %2 = "tf.AddV2"(%0, %cst) : (tensor<8x16xf32>, tensor<f32>) -> tensor<8x16xf32>
  %3 = "tf.Sign"(%2) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %3 : tensor<8x16xf32>

  // CHECK-NEXT: %0 = "lq.Bsign"(%arg0)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_bconv2d
func @fuse_bconv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst = "tf.Const"() { value = dense<[[[[1.0, -1.0], [1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]]]> : tensor<1x2x2x2xf32> } : () -> tensor<1x2x2x2xf32>
  %0 = "lq.Bsign"(%arg0) : (tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Conv2D"(%0, %cst) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x112x112x2xf32>
  return %1 : tensor<1x112x112x2xf32>

  // CHECK: %cst = constant
  // CHECK: %[[post_activation_multiplier:.*]] = constant dense<1.000000e+00> : tensor<2xf32>
  // CHECK: %[[post_activation_bias:.*]] = constant dense<0.000000e+00> : tensor<2xf32>
  // CHECK: %[[output_threshold:.*]] = constant unit
  // CHECK: %[[transpose:.*]] = "tf.Transpose"
  // CHECK-NEXT: %[[conv:.*]] = "lq.Bconv2d"(%arg0, %[[transpose]], %[[post_activation_multiplier]], %[[post_activation_bias]], %[[output_threshold:.*]]) {channels_in = 2 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<1x112x112x2xf32>, tensor<2x1x2x2xf32>, tensor<2xf32>, tensor<2xf32>, none) -> tensor<1x112x112x2xf32>
  // CHECK-NEXT: return %[[conv]]
}

// CHECK-LABEL: @fuse_scaled_bconv2d
func @fuse_scaled_bconv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst = constant dense<[[[[0.3, -0.1], [0.3, 0.1]], [[-0.3, 0.1], [-0.3, 0.1]]]]> : tensor<1x2x2x2xf32>
  %0 = "lq.Bsign"(%arg0) : (tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Conv2D"(%0, %cst) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x112x112x2xf32>
  return %1 : tensor<1x112x112x2xf32>

  // CHECK: %cst = constant
  // CHECK: %[[post_activation_multiplier:.*]] = constant dense<[3.000000e-01, 1.000000e-01]> : tensor<2xf32>
  // CHECK: %[[post_activation_bias:.*]] = constant dense<0.000000e+00> : tensor<2xf32>
  // CHECK: %[[output_threshold:.*]] = constant unit
  // CHECK: %[[transpose:.*]] = "tf.Transpose"
  // CHECK-NEXT: %[[conv:.*]] = "lq.Bconv2d"(%arg0, %[[transpose]], %[[post_activation_multiplier]], %[[post_activation_bias]], %[[output_threshold:.*]]) {channels_in = 2 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "SAME", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<1x112x112x2xf32>, tensor<2x1x2x2xf32>, tensor<2xf32>, tensor<2xf32>, none) -> tensor<1x112x112x2xf32>
  // CHECK-NEXT: return %[[conv]]
}

// CHECK-LABEL: @fuse_dilated_bconv
func @fuse_dilated_bconv(%arg0: tensor<1x128x128x3xf32>) -> tensor<1x128x128x8xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = constant dense<2> : tensor<2x2xi32>
  %cst_1 = constant dense<1.0> : tensor<5x5x3x8xf32>
  %cst_2 = constant unit
  %0 = "lq.Bsign"(%arg0) : (tensor<1x128x128x3xf32>) -> tensor<1x128x128x3xf32>
  %1 = "tf.SpaceToBatchND"(%0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %2 = "tf.Conv2D"(%1, %cst_1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %3 = "tf.BatchToSpaceND"(%2, %cst, %cst_0) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  return %3 : tensor<1x128x128x8xf32>

  // CHECK: %[[transpose:.*]] = "tf.Transpose"
  // CHECK-NEXT: %[[conv:.*]] = "lq.Bconv2d"(%arg0, %[[transpose]], %cst_0, %cst_1, %cst_2) {channels_in = 3 : i32, dilation_height_factor = 2 : i32, dilation_width_factor = 2 : i32, fused_activation_function = "NONE", pad_values = 0 : i32, padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<1x128x128x3xf32>, tensor<8x5x5x3xf32>, tensor<8xf32>, tensor<8xf32>, none) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return %[[conv]] : tensor<1x128x128x8xf32>
}

// CHECK-LABEL: @do_not_fuse_bconv2d
func @do_not_fuse_bconv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst = constant dense<[[[[3.0, -1.0], [0.1, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]]]> : tensor<1x2x2x2xf32>
  %0 = "lq.Bsign"(%arg0) : (tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Conv2D"(%0, %cst) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x112x112x2xf32>
  return %1 : tensor<1x112x112x2xf32>

  // CHECK-NEXT: %cst = constant
  // CHECK-NEXT: %0 = "lq.Bsign"(%arg0)
  // CHECK-NEXT: %1 = "tf.Conv2D"(%0, %cst)
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @do_not_fuse_bconv2d_zero_weight
func @do_not_fuse_bconv2d_zero_weight(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst = constant dense<0.0> : tensor<1x2x2x2xf32>
  %0 = "lq.Bsign"(%arg0) : (tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Conv2D"(%0, %cst) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x112x112x2xf32>
  return %1 : tensor<1x112x112x2xf32>

  // CHECK-NEXT: %cst = constant
  // CHECK-NEXT: %0 = "lq.Bsign"(%arg0)
  // CHECK-NEXT: %1 = "tf.Conv2D"(%0, %cst)
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @fuse_bconv2d_padding
func @fuse_bconv2d_padding(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x16x16x16xf32> {
  %cst = constant dense<1.0> : tensor<3x3x3x16xf32>
  %cst0 = constant dense<1.0> : tensor<f32>
  %cst1 = constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %0 = "lq.Bsign"(%arg0) : (tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  %1 = "tf.PadV2"(%0, %cst1, %cst0) : (tensor<256x32x32x3xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<256x34x34x3xf32>
  %2 = "tf.Conv2D"(%1, %cst) {padding = "VALID", strides = [1, 2, 2, 1]} : (tensor<256x34x34x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x16x16x16xf32>
  return %2 : tensor<256x16x16x16xf32>

  // CHECK:  %[[CST1:.*]] = constant dense<1.000000e+00> : tensor<16xf32>
  // CHECK:  %[[CST2:.*]] = constant dense<0.000000e+00> : tensor<16xf32>
  // CHECK:  %[[CST3:.*]] = constant unit
  // CHECK:  %[[TRP:.*]] = "tf.Transpose"
  // CHECK:  %[[CONV:.*]] = "lq.Bconv2d"(%arg0, %[[TRP]], %[[CST1]], %[[CST2]], %[[CST3:.*]]) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", pad_values = 1 : i32, padding = "SAME", stride_height = 2 : i32, stride_width = 2 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x16x16x16xf32>
}

// CHECK-LABEL: @do_not_fuse_bconv2d_padding_same
func @do_not_fuse_bconv2d_padding_same(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x34x34x16xf32> {
  %cst = constant dense<1.0> : tensor<3x3x3x16xf32>
  %cst0 = constant dense<1.0> : tensor<f32>
  %cst1 = constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %0 = "lq.Bsign"(%arg0) : (tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  %1 = "tf.PadV2"(%0, %cst1, %cst0) : (tensor<256x32x32x3xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<256x34x34x3xf32>
  %2 = "tf.Conv2D"(%1, %cst) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x34x34x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x34x34x16xf32>
  return %2 : tensor<256x34x34x16xf32>

  // CHECK: %0 = "lq.Bsign"
  // CHECK-NEXT: %1 = "tf.PadV2"
  // CHECK-NEXT: %2 = "tf.Conv2D"
}

// CHECK-LABEL: @do_not_fuse_bconv2d_unsupported_constant_padding
func @do_not_fuse_bconv2d_unsupported_constant_padding(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x32x32x16xf32> {
  %cst = constant dense<1.0> : tensor<3x3x3x16xf32>
  %cst0 = constant dense<0.0> : tensor<f32>
  %cst1 = constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %0 = "lq.Bsign"(%arg0) : (tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  %1 = "tf.PadV2"(%0, %cst1, %cst0) : (tensor<256x32x32x3xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<256x34x34x3xf32>
  %2 = "tf.Conv2D"(%1, %cst) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x34x34x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
  return %2 : tensor<256x32x32x16xf32>

  // CHECK: %0 = "lq.Bsign"
  // CHECK-NEXT: %1 = "tf.PadV2"
  // CHECK-NEXT: %2 = "tf.Conv2D"
}

// CHECK-LABEL: @do_not_fuse_bconv2d_padding_wrong_size
func @do_not_fuse_bconv2d_padding_wrong_size(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x36x36x16xf32> {
  %cst = constant dense<1.0> : tensor<3x3x3x16xf32>
  %cst0 = constant dense<1.0> : tensor<f32>
  %cst1 = constant dense<[[0, 0], [2, 2], [2, 2], [0, 0]]> : tensor<4x2xi32>
  %0 = "lq.Bsign"(%arg0) : (tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  %1 = "tf.PadV2"(%0, %cst1, %cst0) : (tensor<256x32x32x3xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<256x36x36x3xf32>
  %2 = "tf.Conv2D"(%1, %cst) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x36x36x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x36x36x16xf32>
  return %2 : tensor<256x36x36x16xf32>

  // CHECK: %0 = "lq.Bsign"
  // CHECK-NEXT: %1 = "tf.PadV2"
  // CHECK-NEXT: %2 = "tf.Conv2D"
}

// CHECK-LABEL: @do_not_fuse_bconv2d_unsymmetric_padding
func @do_not_fuse_bconv2d_unsymmetric_padding(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x32x32x16xf32> {
  %cst = constant dense<1.0> : tensor<3x3x3x16xf32>
  %cst0 = constant dense<1.0> : tensor<f32>
  %cst1 = constant dense<[[0, 0], [2, 0], [2, 0], [0, 0]]> : tensor<4x2xi32>
  %0 = "lq.Bsign"(%arg0) : (tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  %1 = "tf.PadV2"(%0, %cst1, %cst0) : (tensor<256x32x32x3xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<256x34x34x3xf32>
  %2 = "tf.Conv2D"(%1, %cst) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x34x34x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
  return %2 : tensor<256x32x32x16xf32>

  // CHECK: %0 = "lq.Bsign"
  // CHECK-NEXT: %1 = "tf.PadV2"
  // CHECK-NEXT: %2 = "tf.Conv2D"
}

// CHECK-LABEL: @do_not_fuse_bconv2d_non_spatial_padding
func @do_not_fuse_bconv2d_non_spatial_padding(%arg0: tensor<256x32x32x3xf32>) -> tensor<258x32x32x16xf32> {
  %cst = constant dense<1.0> : tensor<3x3x5x16xf32>
  %cst0 = constant dense<1.0> : tensor<f32>
  %cst1 = constant dense<[[1, 1], [1, 1], [1, 1], [1, 1]]> : tensor<4x2xi32>
  %0 = "lq.Bsign"(%arg0) : (tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  %1 = "tf.PadV2"(%0, %cst1, %cst0) : (tensor<256x32x32x3xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<258x34x34x5xf32>
  %2 = "tf.Conv2D"(%1, %cst) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<258x34x34x5xf32>, tensor<3x3x5x16xf32>) -> tensor<258x32x32x16xf32>
  return %2 : tensor<258x32x32x16xf32>

  // CHECK: %0 = "lq.Bsign"
  // CHECK-NEXT: %1 = "tf.PadV2"
  // CHECK-NEXT: %2 = "tf.Conv2D"
}
