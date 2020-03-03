// RUN: lce-tf-opt %s -tfl-optimize-lce | FileCheck %s

// CHECK-LABEL: @fuse_add_into_bconv2d
func @fuse_add_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<1.5> : tensor<16xf32>
  %post_add = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tf.LqceBconv2d64"(%arg0, %arg1, %arg2, %post_add) {data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]>
  // CHECK-NEXT: %0 = "tf.LqceBconv2d64"(%arg0, %arg1, %arg2, %cst)
  // CHECK-NEXT: return %0
}


// CHECK-LABEL: @fuse_sub_into_bconv2d
func @fuse_sub_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<0.5> : tensor<16xf32>
  %post_add = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tf.LqceBconv2d64"(%arg0, %arg1, %arg2, %post_add) {data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = constant dense<[5.000000e-01, 1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01]>
  // CHECK-NEXT: %0 = "tf.LqceBconv2d64"(%arg0, %arg1, %arg2, %cst)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_div_into_bconv2d
func @fuse_div_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<0.5> : tensor<16xf32>
  %post_add = constant dense<1.5> : tensor<16xf32>
  %post_multiply = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tf.LqceBconv2d64"(%arg0, %arg1, %post_multiply, %post_add) {data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.div"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = constant dense<[2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01, 2.400000e+01, 2.600000e+01, 2.800000e+01, 3.000000e+01, 3.200000e+01]>
  // CHECK-NEXT: %cst_0 = constant dense<3.000000e+00>
  // CHECK-NEXT: %0 = "tf.LqceBconv2d64"(%arg0, %arg1, %cst, %cst_0)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_mul_into_bconv2d
func @fuse_mul_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<2.0> : tensor<16xf32>
  %post_add = constant dense<1.5> : tensor<16xf32>
  %post_multiply = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tf.LqceBconv2d64"(%arg0, %arg1, %post_multiply, %post_add) {data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = constant dense<[2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01, 2.400000e+01, 2.600000e+01, 2.800000e+01, 3.000000e+01, 3.200000e+01]>
  // CHECK-NEXT: %cst_0 = constant dense<3.000000e+00>
  // CHECK-NEXT: %0 = "tf.LqceBconv2d64"(%arg0, %arg1, %cst, %cst_0)
  // CHECK-NEXT: return %0
}
