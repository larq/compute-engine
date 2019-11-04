#ifndef COMPUTE_ENGINE_OPS_BCONV2D_SHAPE_INF_H_
#define COMPUTE_ENGINE_OPS_BCONV2D_SHAPE_INF_H_

#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

Status BConv2DShape(shape_inference::InferenceContext* c, int bitwidth);

#endif  // COMPUTE_ENGINE_OPS_BCONV2D_SHAPE_INF_H_
