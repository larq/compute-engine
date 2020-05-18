workspace(name = "larq_compute_engine")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    # This is the "raspberry pi compiler" as used in the tensorflow repository
    # Download size: 250 MB compressed
    # The archive contains GCC versions 4.9.4 (160 MB), 6.5.0 (290 MB) and 8.3.0 (270 MB)
    # Only gcc 6.5.0 is used, the rest is ignored, see the `strip_prefix` parameter.
    name = "arm_compiler",
    build_file = "//third_party:arm_compiler.BUILD",
    sha256 = "b9e7d50ffd9996ed18900d041d362c99473b382c0ae049b2fce3290632d2656f",
    strip_prefix = "rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5/x64-gcc-6.5.0/arm-rpi-linux-gnueabihf/",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz",
        "https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz",
    ],
)

http_archive(
    # This is the latest `aarch64-none-linux-gnu` compiler provided by ARM
    # See https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads
    # The archive contains GCC version 9.2.1
    name = "aarch64_compiler",
    build_file = "//third_party:arm_compiler.BUILD",
    sha256 = "8dfe681531f0bd04fb9c53cf3c0a3368c616aa85d48938eebe2b516376e06a66",
    strip_prefix = "gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu",
    urls = ["https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"],
)

load("//third_party/toolchains/cpus/arm:arm_compiler_configure.bzl", "arm_compiler_configure")

arm_compiler_configure(
    name = "local_config_arm_compiler",
    build_file = "//third_party/toolchains/cpus/arm:BUILD",
    remote_config_repo_aarch64 = "../aarch64_compiler",
    # These paths are relative to some_bazel_dir/external/local_config_arm_compiler/
    remote_config_repo_arm = "../arm_compiler",
)

# To update TensorFlow to a new revision.
# 1. Update the git hash in the urls and the 'strip_prefix' parameter.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | shasum -a 256
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
http_archive(
    name = "org_tensorflow",
    sha256 = "8e86547c8238637cdc895ec53f02adedad4dfa2a7fcfee06dbdeb9f0e3be6168",
    strip_prefix = "tensorflow-a80d96c6634cd005b3841d462030448f9f551a14",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/a80d96c6634cd005b3841d462030448f9f551a14.tar.gz",
    ],
)

# The remainder of this file is derived from (and should be kept in sync with)
# the TensorFlow WORKSPACE file, with the remote caching/execution and
# `py_toolchain` configuration removed:
# https://github.com/tensorflow/tensorflow/blob/master/WORKSPACE

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

# Load tf_repositories() before loading dependencies for other repository so
# that dependencies like com_google_protobuf won't be overridden.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_repositories")

# Please add all new TensorFlow dependencies in workspace.bzl.
tf_repositories()

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

# Use `swift_rules_dependencies` to fetch the toolchains. With the
# `git_repository` rules above, the following call will skip redefining them.
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")

swift_rules_dependencies()

# We must check the bazel version before trying to parse any other BUILD
# files, in case the parsing of those build files depends on the bazel
# version we require here.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

# Keep in sync with `.bazelversion`.
check_bazel_version_at_least("3.0.0")

load("@org_tensorflow//third_party/android:android_configure.bzl", "android_configure")

android_configure(name = "local_config_android")

load("@local_config_android//:android.bzl", "android_workspace")

android_workspace()

# If a target is bound twice, the later one wins, so we have to do tf bindings
# at the end of the WORKSPACE file.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_bind")

tf_bind()

# Required for dependency @com_github_grpc_grpc

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)

apple_support_dependencies()

load("@upb//bazel:repository_defs.bzl", "bazel_version_repository")

bazel_version_repository(name = "bazel_version")

load("@org_tensorflow//third_party/googleapis:repository_rules.bzl", "config_googleapis")

config_googleapis()
