/* Copyright 2019 Plumerai. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "bconv_shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

// Do *not* use a name that already exists within the Tensorflow system
// like `Sign` because it will fail without giving an error message.

// The shape functions can cause crashes when compiled by 'wrong' gcc versions
// https://github.com/tensorflow/tensorflow/issues/29643

REGISTER_OP("LqceBsign")
    .Attr("T: {half, float, double, int8, int32, int64}")
    .Input("input: T")
    .Output("output: T")
    .Doc(
        R"doc(Computes element-wise sign function where 0 is mapped to +1.)doc")
    .SetShapeFn(shape_inference::UnchangedShape);

// The inputs fused_mutiply and fused_add are currently float
// in order to accomodate for batchnorm scales
// Later this might be changed to the int8 system of multipliers+shifts
#define REGISTER_CONV_BITPACKED_OP(OPNAME, BITWIDTH)                                                       \
  REGISTER_OP(OPNAME)                                                                                      \
      .Input("input: T")                                                                                   \
      .Input("filter: T")                                                                                  \
      .Input("fused_multiply: float")                                                                      \
      .Input("fused_add: float")                                                                           \
      .Output("output: T")                                                                                 \
      .Attr("T: {float, double}")                                                                          \
      .Attr("strides: list(int)")                                                                          \
      .Attr(GetPaddingAttrStringWithExplicit())                                                            \
      .Attr(GetExplicitPaddingsAttrString())                                                               \
      .Attr(GetConvnetDataFormatAttrString())                                                              \
      .Attr("filter_format: { 'HWIO' , 'OHWI' , 'OHWI_PACKED' } = 'HWIO' ")                                \
      .Attr("dilations: list(int) = [1, 1, 1, 1]")                                                         \
      .Doc(                                                                                                \
          R"doc(Computes a 2-D binary convolution by binarizing and bitpacking the input and filter.)doc") \
      .SetShapeFn([](shape_inference::InferenceContext* c) {                                               \
        return BConv2DShape(c, BITWIDTH);                                                                  \
      });

REGISTER_CONV_BITPACKED_OP("LqceBconv2d8", 8);
REGISTER_CONV_BITPACKED_OP("LqceBconv2d32", 32);
REGISTER_CONV_BITPACKED_OP("LqceBconv2d64", 64);
