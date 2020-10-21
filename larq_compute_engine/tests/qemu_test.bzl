def lce_qemu_test_suite(
        name,
        platform,
        tests):
    """Test a set of C/C++ binaries using qemu.
      Args:
        name: a unique name for this rule.
        platform: either "arm32" or "aarch64"
        tests: list of cc_test targets
      """
    if platform == "arm32":
        src = "//larq_compute_engine/tests:test_arm32_binary.sh"
        qemu_data = "@arm_compiler//:compiler_pieces"
    elif platform == "aarch64":
        src = "//larq_compute_engine/tests:test_aarch64_binary.sh"
        qemu_data = "@aarch64_compiler//:aarch64_compiler_pieces"
    else:
        fail("Invalid platform name in lce_qemu_test_suite", platform)

    sh_tests = []
    for test in tests:
        # `test` is a Bazel target name
        # From this we extract a path to the compiled binary
        test_path = test
        if test_path.startswith("//"):
            test_path = test_path[2:]
        else:
            test_path = native.package_name() + "/" + test_path
        test_path = test_path.replace(":", "/")

        # We also have to create a unique identifier for this sh_test target
        test_suffix = test.split(":", None)[-1]
        sh_name = name + "_" + test_suffix

        # Finally create a new sh_test target
        native.sh_test(
            name = sh_name,
            size = "large",
            srcs = [src],
            args = [test_path],
            data = [test, qemu_data],
            shard_count = 2,
        )

        # And add that sh_test target to the list
        sh_tests = sh_tests + [sh_name]

    # Collect the newly created targets in a regular test_suite
    native.test_suite(name = name, tests = sh_tests)
