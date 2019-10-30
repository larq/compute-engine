load("//ext/tf:tf_configure.bzl", "tf_configure")
load("//ext/gpu:cuda_configure.bzl", "cuda_configure")

tf_configure(name = "local_config_tf")

cuda_configure(name = "local_config_cuda")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "googletest",
    url = "https://github.com/google/googletest/archive/release-1.10.0.zip",
    sha256 = "94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91",
    build_file = "@//ext:gtest.BUILD",
    strip_prefix = "googletest-release-1.10.0",
)

http_archive(
    name = "eigen_archive",
    build_file = "@//ext:eigen.BUILD",
    sha256 = "a126a1af9ec3b3f646c4896bf69a4bb71e9ebfb30c50c3182f01270a704a4093",
    strip_prefix = "eigen-eigen-89abeb806e2e",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/bitbucket.org/eigen/eigen/get/89abeb806e2e.tar.gz",
        "https://bitbucket.org/eigen/eigen/get/89abeb806e2e.tar.gz",
    ],
)

