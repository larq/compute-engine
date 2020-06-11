// RUN: lce-tf-opt %s -tfl-legalize-lce | FileCheck %s

// CHECK-LABEL: @legalize_bconv2d
func @legalize_bconv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: none) -> tensor<256x30x30x16xf32> {
  %0 = "lq.Bconv2d"(%arg0, %arg1, %arg2, %arg3, %arg4) {channels_in = 3 : i32, dilation_height_factor = 1 : i32, dilation_width_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_height = 1 : i32, stride_width = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tfl.custom"(%arg0, %arg1, %arg2, %arg3, %arg4) {custom_code = "LceBconv2d", custom_option = opaque<"lq", "0x6368616E6E656C735F696E0064696C6174696F6E5F6865696768745F666163746F720064696C6174696F6E5F77696474685F666163746F720066757365645F61637469766174696F6E5F66756E6374696F6E00044E4F4E45007061645F76616C7565730070616464696E67000556414C4944007374726964655F686569676874007374726964655F776964746800088F846E593A30221508010803010149003201010404041404140404102401"> : tensor<173xi8>} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, none) -> tensor<256x30x30x16xf32>
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @legalize_bmax_pool2d
func @legalize_bmax_pool2d(%arg0: tensor<256x32x32x3xi32>) -> tensor<256x16x16x3xi32> {
  %0 = "lq.BMaxPool2d"(%arg0) {filter_height = 2 : i32, filter_width = 2 : i32, padding = "SAME", stride_height = 2 : i32, stride_width = 2 : i32} : (tensor<256x32x32x3xi32>) -> tensor<256x16x16x3xi32>
  return %0 : tensor<256x16x16x3xi32>

  // CHECK: %0 = "tfl.custom"(%arg0) {custom_code = "LceBMaxPool2d", custom_option = opaque<"lq", "0x70616464696E67000453414D45007374726964655F7769647468007374726964655F6865696768740066696C7465725F77696474680066696C7465725F68656967687400050F1D472D3B050105020246020204041404040A2401"> : tensor<90xi8>} : (tensor<256x32x32x3xi32>) -> tensor<256x16x16x3xi32>
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @legalize_bsign
func @legalize_bsign(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32> {
  %0 = "lq.Bsign"(%arg0) {} : (tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  return %0 : tensor<256x32x32x3xf32>

  // CHECK: %0 = "tfl.custom"(%arg0) {custom_code = "LceBsign", custom_option = opaque<"lq", "0x"> : tensor<0xi8>} : (tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  // CHECK-NEXT: return %0
}
