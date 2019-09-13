load("//tf:tf_configure.bzl", "tf_configure")
load("//gpu:cuda_configure.bzl", "cuda_configure")

tf_configure(name = "local_config_tf")

cuda_configure(name = "local_config_cuda")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "@//:gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
)