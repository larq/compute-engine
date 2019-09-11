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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// Do *not* use a name that already exists within the Tensorflow system
// like `Sign` because it will fail without giving an error message.

// The shape functions cause crashes. This might be caused by gcc versions
// https://github.com/tensorflow/tensorflow/issues/29643

REGISTER_OP("Bgemm")
    .Attr("T: {float, double, int32}")
    .Input("input_a: T")
    .Input("input_b: T")
    .Output("output_c: T");

REGISTER_OP("FastSign")
    .Attr("T: {half, float, double, int8, int32, int64}")
    .Input("input: T")
    .Output("output: T");
    //.SetShapeFn(shape_inference::UnchangedShape);

