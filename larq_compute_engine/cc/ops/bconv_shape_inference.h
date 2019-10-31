#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

Status BConv2DShape(shape_inference::InferenceContext* c, int bitwidth);
