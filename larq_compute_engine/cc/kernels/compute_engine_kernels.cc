/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

template <typename T>
class BgemmOp : public OpKernel {
 public:
  explicit BgemmOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& a_tensor = context->input(0);
    const Tensor& b_tensor = context->input(1);

    // Matirx-Matrix multiplication, hence there should be 2 dimensions.
    OP_REQUIRES(context, a_tensor.dims() == 2,
                errors::InvalidArgument("input must be 2-dimensional",
                                        a_tensor.shape().DebugString()));

    OP_REQUIRES(context, b_tensor.dims() == 2,
                errors::InvalidArgument("input must be 2-dimensional: ",
                                        b_tensor.shape().DebugString()));

    // number of the first matrix columns should be same as
    // the number of second matrix rows
    OP_REQUIRES(context, a_tensor.dim_size(1) == b_tensor.dim_size(0),
                errors::InvalidArgument(
                    "number of the first matrix cols and second matrix rows must be the same: ",
                    a_tensor.dim_size(1), " vs ", b_tensor.dim_size(0)));

    const int m = a_tensor.dim_size(0);
    const int n = b_tensor.dim_size(1);
    const int k = a_tensor.dim_size(1);

    // Create an output tensor
    const auto output_shape = TensorShape({m, n});

    // if there is nothing to compute, return.
    if (output_shape.num_elements() == 0) {
      return;
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    // Set all elements of the output tensor to 0.
    auto output_flat = output_tensor->flat<T>();
    const int N = output_flat.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }
  }
};


#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Bgemm").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BgemmOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
