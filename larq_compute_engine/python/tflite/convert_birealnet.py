"""Example of converting BiRealNet to TF lite."""
import larq_compute_engine as lqce
from larq_zoo import BiRealNet

model = BiRealNet()

conv = lqce.ModelConverter(model)
conv.convert("/tmp/birealnet.tflite")
