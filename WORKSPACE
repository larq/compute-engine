workspace(name = "larq_compute_engine")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# To update TensorFlow to a new revision.
# 1. Update the git hash in the urls and the 'strip_prefix' parameter.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | shasum -a 256
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
http_archive(
    name = "org_tensorflow",
    patch_args = ["-p1"],
    patch_tool = "patch",
    patches = [
        "//third_party/tensorflow_patches:disable_forced_mkl.patch",
        "//third_party/tensorflow_patches:fix_armhf_xnnpack.patch",
    ],
    sha256 = "263f747102e531dc52e5c912db247ab6053070bb1f549294aca6f5d696529128",
    strip_prefix = "tensorflow-8727d035e7aa593720d16a5f57f70f3b5a93bd00",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/8727d035e7aa593720d16a5f57f70f3b5a93bd00.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()
