// RUN: lce-tf-opt %s -tfl-optimize-lce | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: @fuse_add_into_bconv2d
func @fuse_add_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<1.5> : tensor<16xf32>
  %post_activation_bias = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]>
  // CHECK-NEXT: %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %cst)
  // CHECK-NEXT: return %0
}


// CHECK-LABEL: @fuse_sub_into_bconv2d
func @fuse_sub_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<0.5> : tensor<16xf32>
  %post_activation_bias = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = constant dense<[5.000000e-01, 1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01]>
  // CHECK-NEXT: %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %cst)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_div_into_bconv2d
func @fuse_div_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<0.5> : tensor<16xf32>
  %post_activation_bias = constant dense<1.5> : tensor<16xf32>
  %post_activation_multiplier = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.div"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = constant dense<[2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01, 2.400000e+01, 2.600000e+01, 2.800000e+01, 3.000000e+01, 3.200000e+01]>
  // CHECK-NEXT: %cst_0 = constant dense<3.000000e+00>
  // CHECK-NEXT: %0 = "tf.LceBconv2d"(%arg0, %arg1, %cst, %cst_0)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_mul_into_bconv2d
func @fuse_mul_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<2.0> : tensor<16xf32>
  %post_activation_bias = constant dense<1.5> : tensor<16xf32>
  %post_activation_multiplier = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", padding = "SAME", read_bipacked_input = false, strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK-NEXT: %cst = constant dense<[2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01, 2.400000e+01, 2.600000e+01, 2.800000e+01, 3.000000e+01, 3.200000e+01]>
  // CHECK-NEXT: %cst_0 = constant dense<3.000000e+00>
  // CHECK-NEXT: %0 = "tf.LceBconv2d"(%arg0, %arg1, %cst, %cst_0)
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_relu_into_bconv2d
func @fuse_relu_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = constant dense<0.0> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", read_bipacked_input = false, strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %cst, %cst_0) {activation = "RELU", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]}
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_relu6_into_bconv2d
func @fuse_relu6_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = constant dense<0.0> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", read_bipacked_input = false, strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu6"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %cst, %cst_0) {activation = "RELU6", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]}
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_relu1_into_bconv2d
func @fuse_relu1_into_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = constant dense<0.0> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", read_bipacked_input = false, strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu_n1_to_1"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %cst, %cst_0) {activation = "RELU_N1_TO_1", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]}
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fuse_relu_into_bconv2d_padding_same
func @fuse_relu_into_bconv2d_padding_same(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x32x32x16xf32> {
  %post_activation_multiplier = constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = constant dense<0.0> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 1 : i32, padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x32x32x16xf32>) -> tensor<256x32x32x16xf32>
  return %1 : tensor<256x32x32x16xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %cst, %cst_0) {activation = "RELU", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 1 : i32, padding = "SAME", strides = [1, 1, 1, 1]}
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @do_not_fuse_relu_into_bconv2d_padding_same
func @do_not_fuse_relu_into_bconv2d_padding_same(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x32x32x16xf32> {
  %post_activation_multiplier = constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = constant dense<0.0> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x32x32x16xf32>) -> tensor<256x32x32x16xf32>
  return %1 : tensor<256x32x32x16xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %cst, %cst_0) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "SAME", strides = [1, 1, 1, 1]}
  // CHECK-NEXT: %1 = "tfl.relu"(%0)
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @do_not_fuse_relu_into_bconv2d_no_post_activation_bias
func @do_not_fuse_relu_into_bconv2d_no_post_activation_bias(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = constant dense<1.0> : tensor<16xf32>
  %post_activation_bias = constant dense<5.0> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %cst, %cst_0) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]}
  // CHECK-NEXT: %1 = "tfl.relu"(%0)
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @do_not_fuse_relu_into_bconv2d_no_post_activation_multiplier
func @do_not_fuse_relu_into_bconv2d_no_post_activation_multiplier(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %post_activation_multiplier = constant dense<0.8> : tensor<16xf32>
  %post_activation_bias = constant dense<0.0> : tensor<16xf32>
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %post_activation_multiplier, %post_activation_bias) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %cst, %cst_0) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]}
  // CHECK-NEXT: %1 = "tfl.relu"(%0)
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @bitpack_bconv2d_filters
func @bitpack_bconv2d_filters(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<1.0> : tensor<16x3x3x3xf32>
  %0 = "tf.LceBconv2d"(%arg0, %cst, %arg1, %arg2) {activation = "NONE", channels_in = 3 : i32, filter_format = "OHWI", padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<0> : tensor<16x3x3x1xi32>
  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %cst, %arg1, %arg2) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI_PACKED", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<16x3x3x1xi32>, tensor<16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @bitpack_activations_between_two_bconv2ds_valid_padding
func @bitpack_activations_between_two_bconv2ds_valid_padding(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<65x3x3x3xf32>, %arg2: tensor<65xf32>, %arg3: tensor<65xf32>, %arg4: tensor<8x3x3x65xf32>, %arg5: tensor<8xf32>, %arg6: tensor<8xf32>) -> tensor<256x28x28x8xf32> {
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>) -> tensor<256x30x30x65xf32>
  %1 = "tf.LceBconv2d"(%0, %arg4, %arg5, %arg6) {activation = "NONE", channels_in = 65 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x30x30x65xf32>, tensor<8x3x3x65xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<256x28x28x8xf32>
  return %1 : tensor<256x28x28x8xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>) -> tensor<256x30x30x3xi32>
  // CHECK-NEXT: %1 = "tf.LceBconv2d"(%0, %arg4, %arg5, %arg6) {activation = "NONE", channels_in = 65 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x30x30x3xi32>, tensor<8x3x3x65xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<256x28x28x8xf32>
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @bitpack_activations_between_two_bconv2ds_same_one_padding
func @bitpack_activations_between_two_bconv2ds_same_one_padding(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<65x3x3x3xf32>, %arg2: tensor<65xf32>, %arg3: tensor<65xf32>, %arg4: tensor<8x3x3x65xf32>, %arg5: tensor<8xf32>, %arg6: tensor<8xf32>) -> tensor<256x30x30x8xf32> {
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 1 : i32, padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>) -> tensor<256x32x32x65xf32>
  %1 = "tf.LceBconv2d"(%0, %arg4, %arg5, %arg6) {activation = "NONE", channels_in = 65 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x65xf32>, tensor<8x3x3x65xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<256x30x30x8xf32>
  return %1 : tensor<256x30x30x8xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 1 : i32, padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>) -> tensor<256x32x32x3xi32>
  // CHECK-NEXT: %1 = "tf.LceBconv2d"(%0, %arg4, %arg5, %arg6) {activation = "NONE", channels_in = 65 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xi32>, tensor<8x3x3x65xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<256x30x30x8xf32>
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @do_not_bitpack_activations_between_two_bconv2ds_same_zero_padding
func @do_not_bitpack_activations_between_two_bconv2ds_same_zero_padding(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<65x3x3x3xf32>, %arg2: tensor<65xf32>, %arg3: tensor<65xf32>, %arg4: tensor<8x3x3x65xf32>, %arg5: tensor<8xf32>, %arg6: tensor<8xf32>) -> tensor<256x30x30x8xf32> {
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>) -> tensor<256x32x32x65xf32>
  %1 = "tf.LceBconv2d"(%0, %arg4, %arg5, %arg6) {activation = "NONE", channels_in = 65 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x65xf32>, tensor<8x3x3x65xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<256x30x30x8xf32>
  return %1 : tensor<256x30x30x8xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>) -> tensor<256x32x32x65xf32>
  // CHECK-NEXT: %1 = "tf.LceBconv2d"(%0, %arg4, %arg5, %arg6) {activation = "NONE", channels_in = 65 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x65xf32>, tensor<8x3x3x65xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<256x30x30x8xf32>
  // CHECK-NEXT: return %1
}

// CHECK-LABEL: @do_not_bitpack_activations_between_two_bconv2ds_same_one_padding_multiple_uses
func @do_not_bitpack_activations_between_two_bconv2ds_same_one_padding_multiple_uses(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<65x3x3x3xf32>, %arg2: tensor<65xf32>, %arg3: tensor<65xf32>, %arg4: tensor<8x3x3x65xf32>, %arg5: tensor<8xf32>, %arg6: tensor<8xf32>) -> (tensor<256x32x32x65xf32>, tensor<256x30x30x8xf32>) {
  %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 1 : i32, padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>) -> tensor<256x32x32x65xf32>
  %1 = "tf.LceBconv2d"(%0, %arg4, %arg5, %arg6) {activation = "NONE", channels_in = 65 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x65xf32>, tensor<8x3x3x65xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<256x30x30x8xf32>
  return %0, %1: tensor<256x32x32x65xf32>, tensor<256x30x30x8xf32>

  // CHECK: %0 = "tf.LceBconv2d"(%arg0, %arg1, %arg2, %arg3) {activation = "NONE", channels_in = 3 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 1 : i32, padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<65x3x3x3xf32>, tensor<65xf32>, tensor<65xf32>) -> tensor<256x32x32x65xf32>
  // CHECK-NEXT: %1 = "tf.LceBconv2d"(%0, %arg4, %arg5, %arg6) {activation = "NONE", channels_in = 65 : i32, data_format = "NHWC", dilations = [1, 1, 1, 1], filter_format = "OHWI", pad_values = 0 : i32, padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x65xf32>, tensor<8x3x3x65xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<256x30x30x8xf32>
  // CHECK-NEXT: return %0, %1
}
