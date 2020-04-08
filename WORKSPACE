workspace(name = "larq_compute_engine")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "googlebenchmark",
    sha256 = "2d22dd3758afee43842bb504af1a8385cccb3ee1f164824e4837c1c1b04d92a0",
    strip_prefix = "benchmark-1.5.0",
    url = "https://github.com/google/benchmark/archive/v1.5.0.zip",
)

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
    # This is the latest `aarch64-linux-gnu` compiler provided by ARM
    # See https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads
    # Download size: 260 MB compressed, 1.5 GB uncompressed
    # The archive contains GCC version 8.3.0
    name = "aarch64_compiler",
    build_file = "//third_party:aarch64_compiler.BUILD",
    sha256 = "8ce3e7688a47d8cd2d8e8323f147104ae1c8139520eca50ccf8a7fa933002731",
    strip_prefix = "gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/",
    url = "https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz?revision=2e88a73f-d233-4f96-b1f4-d8b36e9bb0b9&la=en&hash=167687FADA00B73D20EED2A67D0939A197504ACD",
)

load("//third_party/toolchains/cpus/arm:arm_compiler_configure.bzl", "arm_compiler_configure")

arm_compiler_configure(
    name = "local_config_arm_compiler",
    build_file = "//third_party/toolchains/cpus/arm:BUILD",
    remote_config_repo_aarch64 = "../aarch64_compiler",
    #These paths are relative to some_bazel_dir/external/local_config_arm_compiler/
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
    sha256 = "902166e31150f341b0f993fd375ef7e6021a1183bdf96beefdbad0a3d80738f6",
    strip_prefix = "tensorflow-667818621b6ddb439df945e28bb0d21d3d1f019d",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/667818621b6ddb439df945e28bb0d21d3d1f019d.tar.gz",
    ],
)

# START: Upstream TensorFlow dependencies
# TensorFlow build depends on these dependencies.
# Needs to be in-sync with TensorFlow sources.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)  # https://github.com/bazelbuild/bazel-skylib/releases
# END: Upstream TensorFlow dependencies

# Apple and Swift rules.
http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "a045a436b642c70fb0c10ca84ff0fd2dcbd59cc89100d597a61e8374afafb366",
    urls = ["https://github.com/bazelbuild/rules_apple/releases/download/0.18.0/rules_apple.0.18.0.tar.gz"],
)  # https://github.com/bazelbuild/rules_apple/releases

http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "18cd4df4e410b0439a4935f9ca035bd979993d42372ba79e7f2d4fafe9596ef0",
    urls = ["https://github.com/bazelbuild/rules_swift/releases/download/0.12.1/rules_swift.0.12.1.tar.gz"],
)  # https://github.com/bazelbuild/rules_swift/releases

http_archive(
    name = "build_bazel_apple_support",
    sha256 = "122ebf7fe7d1c8e938af6aeaee0efe788a3a2449ece5a8d6a428cb18d6f88033",
    urls = ["https://github.com/bazelbuild/apple_support/releases/download/0.7.1/apple_support.0.7.1.tar.gz"],
)  # https://github.com/bazelbuild/apple_support/releases

# Import all of the tensorflow dependencies.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

# Bootstrap TensorFlow deps last so that ours can take precedence.
tf_workspace(tf_repo_name = "org_tensorflow")

# android bazel configurations
load("@org_tensorflow//third_party/android:android_configure.bzl", "android_configure")

android_configure(name = "local_config_android")

load("@local_config_android//:android.bzl", "android_workspace")

android_workspace()
