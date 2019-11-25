load("//ext/tf:tf_configure.bzl", "tf_configure")
tf_configure(name = "local_config_tf")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "googletest",
    url = "https://github.com/google/googletest/archive/release-1.10.0.zip",
    sha256 = "94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91",
    build_file = "@//ext:gtest.BUILD",
    strip_prefix = "googletest-release-1.10.0",
)

http_archive(
    name = "googlebenchmark",
    url = "https://github.com/google/benchmark/archive/v1.5.0.zip",
    sha256 = "2d22dd3758afee43842bb504af1a8385cccb3ee1f164824e4837c1c1b04d92a0",
    strip_prefix = "benchmark-1.5.0",
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

http_archive(
    # This is the "raspberry pi compiler" as used in the tensorflow repository
    # Download size: 250 MB compressed
    # The archive contains GCC versions 4.9.4 (160 MB), 6.5.0 (290 MB) and 8.3.0 (270 MB)
    # Only gcc 6.5.0 is used, the rest is ignored, see the `strip_prefix` parameter.
    name = "arm_compiler",
    build_file = "//ext:arm_compiler.BUILD",
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
    build_file = "//ext:aarch64_compiler.BUILD",
    sha256 = "8ce3e7688a47d8cd2d8e8323f147104ae1c8139520eca50ccf8a7fa933002731",
    strip_prefix = "gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/",
    url = "https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz?revision=2e88a73f-d233-4f96-b1f4-d8b36e9bb0b9&la=en&hash=167687FADA00B73D20EED2A67D0939A197504ACD",
)

load("//ext/toolchains/cpus/arm:arm_compiler_configure.bzl", "arm_compiler_configure")

arm_compiler_configure(
    name = "local_config_arm_compiler",
    build_file = "//ext/toolchains/cpus/arm:BUILD",
    #These paths are relative to some_bazel_dir/external/local_config_arm_compiler/
    remote_config_repo_arm = "../arm_compiler",
    remote_config_repo_aarch64 = "../aarch64_compiler",
)

# To update TensorFlow to a new revision.
# 1. Update the git hash in the urls and the 'strip_prefix' parameter.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
http_archive(
    name = "org_tensorflow",
    sha256 = "bdbe31d6de69964e364612de466b4624b292788988f95c5d27dabdc339fe50f1",
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/597a30bc61134ee1deec0b439b3649f346f1f119.tar.gz",
        "https://github.com/tensorflow/tensorflow/archive/597a30bc61134ee1deec0b439b3649f346f1f119.tar.gz",
    ],
    strip_prefix = "tensorflow-597a30bc61134ee1deec0b439b3649f346f1f119",
)

# load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
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
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)  # https://github.com/bazelbuild/bazel-skylib/releases
# END: Upstream TensorFlow dependencies

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")
