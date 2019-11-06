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
    # Note: this is ~250 MB
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
    # Note: this is ~260 MB
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
    remote_config_repo = "../arm_compiler", #This path is relative to some_bazel_dir/external/local_config_arm_compiler/
)

