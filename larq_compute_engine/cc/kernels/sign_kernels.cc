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
#include <cmath>

using namespace tensorflow;

// Here we will provide architecture-specific implementations
namespace compute_engine {
namespace detail {

// This is the generic template
// It can be specialized when needed
template <typename T>
T sign(T x) {
  return (x >= 0 ? T(1) : T(-1));
}

// Specialization for float
template <>
float sign(float x) {
  return (std::signbit(x) ? -1.0f : 1.0f);
}

}  // namespace detail
}  // namespace compute_engine

template <typename T>
class SignOp : public OpKernel {
 public:
  explicit SignOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get the input tensor
    const Tensor& in_tensor = context->input(0);

    // Create an output tensor
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, in_tensor.shape(), &out_tensor));

    // Get flat representations of the tensor
    auto input_flat = in_tensor.flat<T>();
    auto output_flat = out_tensor->flat<T>();

    // Compute the sign
    const int N = input_flat.size();
    for (int i = 0; i < N; ++i) {
      output_flat(i) = compute_engine::detail::sign<T>(input_flat(i));
    }
  }
};

#define REGISTER_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(    \
      Name("FastSign").Device(DEVICE_CPU).TypeConstraint<type>("T"), SignOp<type>)

REGISTER_KERNEL(int8);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL
