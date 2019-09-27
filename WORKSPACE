load("//external/tf:tf_configure.bzl", "tf_configure")
load("//external/gpu:cuda_configure.bzl", "cuda_configure")

tf_configure(name = "local_config_tf")

cuda_configure(name = "local_config_cuda")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

new_git_repository(
    name = "googletest",
    build_file = "gtest.BUILD",
    remote = "https://github.com/google/googletest",
    tag = "release-1.8.0",
)