CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python3

COMPUTE_ENGINE_SRCS = $(wildcard larq_compute_engine/cc/kernels/*.cc) $(wildcard larq_compute_engine/cc/kernels/*.h) $(wildcard larq_compute_engine/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

COMPUTE_ENGINE_TARGET_LIB = larq_compute_engine/python/ops/_larq_compute_engine_ops.so

# compute engine ops for CPU
compute_engine_ops: $(COMPUTE_ENGINE_TARGET_LIB)

$(COMPUTE_ENGINE_TARGET_LIB): $(COMPUTE_ENGINE_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

compute_engine_pip_pkg: $(COMPUTE_ENGINE_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

compute_engine_test: larq_compute_engine/python/ops/compute_engine_ops_test.py larq_compute_engine/python/ops/compute_engine_ops.py $(COMPUTE_ENGINE_TARGET_LIB)
		$(PYTHON_BIN_PATH) -m pytest -vs larq_compute_engine/python/

clean:
	rm -f $(COMPUTE_ENGINE_TARGET_LIB)
