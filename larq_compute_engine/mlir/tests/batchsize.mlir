// RUN: lce-tf-opt %s -mlir-setbatchsize -verify-diagnostics | FileCheck %s

// This is an IR dump from the following simple 2-input model
//    img1 = tf.keras.layers.Input(shape=(4,))
//    img2 = tf.keras.layers.Input(shape=(6,))
//    x = tf.keras.layers.Dense(6)(img1) + img2
//    return tf.keras.Model([img1, img2], x)
// Both inputs have a dynamic batch size

// CHECK-LABEL: @dual_input_model
func @dual_input_model(%arg0: tensor<?x6xf32> {tf_saved_model.index_path = ["input_2"]}, %arg1: tensor<?x4xf32> {tf_saved_model.index_path = ["input_1"]}, %arg2: tensor<!tf.resource<tensor<6xf32>>> {tf_saved_model.bound_input = @"dense/bias"}, %arg3: tensor<!tf.resource<tensor<4x6xf32>>> {tf_saved_model.bound_input = @"dense/kernel"}) -> (tensor<?x6xf32> {tf_saved_model.index_path = ["tf.__operators__.add"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_2:0,serving_default_input_1:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  %0 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf.resource<tensor<6xf32>>>) -> tensor<6xf32>
  %1 = "tf.ReadVariableOp"(%arg3) {device = ""} : (tensor<!tf.resource<tensor<4x6xf32>>>) -> tensor<4x6xf32>
  %2 = "tf.MatMul"(%arg1, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x6xf32>) -> tensor<?x6xf32>
  %3 = "tf.BiasAdd"(%2, %0) {data_format = "NHWC", device = ""} : (tensor<?x6xf32>, tensor<6xf32>) -> tensor<?x6xf32>
  %4 = "tf.AddV2"(%3, %arg0) {device = ""} : (tensor<?x6xf32>, tensor<?x6xf32>) -> tensor<?x6xf32>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<?x6xf32>) -> tensor<?x6xf32>
  %6 = "tf.Identity"(%5) {device = ""} : (tensor<?x6xf32>) -> tensor<?x6xf32>
  return %6 : tensor<?x6xf32>
  // CHECK: %arg0: tensor<1x6xf32>
  // CHECK: %arg1: tensor<1x4xf32>
}

// This is the same model, but one of the two inputs has been given a fixed batch size in Python

// CHECK-LABEL: @dual_input_one_fixed_size
func @dual_input_one_fixed_size(%arg0: tensor<?x6xf32> {tf_saved_model.index_path = ["input_2"]}, %arg1: tensor<1x4xf32> {tf_saved_model.index_path = ["input_1"]}, %arg2: tensor<!tf.resource<tensor<6xf32>>> {tf_saved_model.bound_input = @"dense/bias"}, %arg3: tensor<!tf.resource<tensor<4x6xf32>>> {tf_saved_model.bound_input = @"dense/kernel"}) -> (tensor<?x6xf32> {tf_saved_model.index_path = ["tf.__operators__.add"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_2:0,serving_default_input_1:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  %0 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf.resource<tensor<6xf32>>>) -> tensor<6xf32>
  %1 = "tf.ReadVariableOp"(%arg3) {device = ""} : (tensor<!tf.resource<tensor<4x6xf32>>>) -> tensor<4x6xf32>
  %2 = "tf.MatMul"(%arg1, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x4xf32>, tensor<4x6xf32>) -> tensor<1x6xf32>
  %3 = "tf.BiasAdd"(%2, %0) {data_format = "NHWC", device = ""} : (tensor<1x6xf32>, tensor<6xf32>) -> tensor<1x6xf32>
  %4 = "tf.AddV2"(%3, %arg0) {device = ""} : (tensor<1x6xf32>, tensor<?x6xf32>) -> tensor<?x6xf32>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<?x6xf32>) -> tensor<?x6xf32>
  %6 = "tf.Identity"(%5) {device = ""} : (tensor<?x6xf32>) -> tensor<?x6xf32>
  return %6 : tensor<?x6xf32>
  // CHECK: %arg0: tensor<1x6xf32>
  // CHECK: %arg1: tensor<1x4xf32>
}