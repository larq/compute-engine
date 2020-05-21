// RUN: lce-tf-opt %s -lce-op-removal-tf | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: @snapshot
func @snapshot(%arg0: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "tf.Snapshot"(%arg0) : (tensor<3xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
  // Should be converted to Identity and then from Identity to value
  // CHECK-NEXT: return %arg0 : tensor<3xi32>
}

// CHECK-LABEL: @stop_gradient
func @stop_gradient(%arg0: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "tf.StopGradient"(%arg0) : (tensor<3xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
  // Should be converted to Identity and then from Identity to value
  // CHECK-NEXT: return %arg0 : tensor<3xi32>
}

// CHECK-LABEL: @check_numerics
func @check_numerics(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "tf.CheckNumerics"(%arg0) {message = ""}: (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
  // Should be converted to Identity and then from Identity to value
  // CHECK-NEXT: return %arg0 : tensor<3xf32>
}

// CHECK-LABEL: @placeholder_with_default
func @placeholder_with_default(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "tf.PlaceholderWithDefault"(%arg0): (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
  // Should be converted to Identity and then from Identity to value
  // CHECK-NEXT: return %arg0 : tensor<3xf32>
}

// CHECK-LABEL: @identity
func @identity(%arg0: tensor<10xi32>, %arg1: tensor<20xi32>, %arg2: tensor<30xi32>) -> (tensor<10xi32>, tensor<20xi32>, tensor<30xi32>) {
  %0 = "tf.Identity"(%arg0) : (tensor<10xi32>) -> tensor<10xi32>
  %1:2 = "tf.IdentityN"(%arg1,%arg2) : (tensor<20xi32>, tensor<30xi32>) -> (tensor<20xi32>, tensor<30xi32>)
  return %0, %1#0, %1#1: tensor<10xi32>, tensor<20xi32>, tensor<30xi32>

  // CHECK-NEXT: return %arg0, %arg1, %arg2
}

// CHECK-LABEL: @unused_bsign
func @unused_bsign(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "tf.LceBsign"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
  return %arg0 : tensor<10xf32>

  // CHECK-NEXT: return %arg0 : tensor<10xf32>
}

// CHECK-LABEL: @used_bsign
func @used_bsign(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "tf.LceBsign"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>

  // CHECK-NEXT: %0 = "tf.LceBsign"
}

// CHECK-LABEL: @unused_bconv2d
func @unused_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>) -> tensor<256x32x32x3xf32> {
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  return %arg0 : tensor<256x32x32x3xf32>

  // CHECK-NEXT: return %arg0 : tensor<256x32x32x3xf32>
}

// CHECK-LABEL: @used_bconv2d
func @used_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>) -> tensor<256x32x32x16xf32> {
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  return %0 : tensor<256x32x32x16xf32>

  // CHECK-NEXT: %0 = "tf.LceBconv2d"
}

// CHECK-LABEL: @unused_bmaxpool2d
func @unused_bmaxpool2d(%arg0: tensor<256x32x32x3xi32>) -> tensor<256x32x32x3xi32> {
  %0 = "tf.LceBMaxPool2d"(%arg0) {filter_height = 2 : i32, filter_width = 2 : i32, padding = "SAME", stride_height = 2 : i32, stride_width = 2 : i32} : (tensor<256x32x32x3xi32>) -> tensor<256x16x16x3xi32>
  return %arg0 : tensor<256x32x32x3xi32>

  // CHECK-NEXT: return %arg0 : tensor<256x32x32x3xi32>
}

// CHECK-LABEL: @used_bmaxpool2d
func @used_bmaxpool2d(%arg0: tensor<256x32x32x3xi32>) -> tensor<256x16x16x3xi32> {
  %0 = "tf.LceBMaxPool2d"(%arg0) {filter_height = 2 : i32, filter_width = 2 : i32, padding = "SAME", stride_height = 2 : i32, stride_width = 2 : i32} : (tensor<256x32x32x3xi32>) -> tensor<256x16x16x3xi32>
  return %0 : tensor<256x16x16x3xi32>

  // CHECK-NEXT: %0 = "tf.LceBMaxPool2d"
}
