#!/usr/bin/env python

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2021 Larq Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import platform
import subprocess
import sys

_LCE_BAZELRC = ".lce_configure.bazelrc"


def is_windows():
    return platform.system() == "Windows"


def is_linux():
    return platform.system() == "Linux"


def is_macos():
    return platform.system() == "Darwin"


def is_ppc64le():
    return platform.machine() == "ppc64le"


def is_cygwin():
    return platform.system().startswith("CYGWIN_NT")


def write_to_bazelrc(line):
    with open(_LCE_BAZELRC, "a") as f:
        f.write(line + "\n")


def write_action_env_to_bazelrc(var_name, var):
    write_to_bazelrc('build --action_env {}="{}"'.format(var_name, str(var)))


def get_input(question):
    try:
        answer = input(question)
    except EOFError:
        answer = ""
    return answer


def get_from_env_or_user_or_default(environ_cp, var_name, ask_for_var, var_default):
    """Get var_name either from env, or user or default.
    If var_name has been set as environment variable, use the preset value, else
    ask for user input. If no input is provided, the default is used.
    Args:
      environ_cp: copy of the os.environ.
      var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
      ask_for_var: string for how to ask for user input.
      var_default: default value string.
    Returns:
      string value for var_name
    """
    var = environ_cp.get(var_name)
    if not var:
        var = get_input(ask_for_var)
        print("\n")
    if not var:
        var = var_default
    return var


def get_var(
    environ_cp,
    var_name,
    query_item,
    enabled_by_default,
    question=None,
    yes_reply=None,
    no_reply=None,
):
    """Get boolean input from user.
    If var_name is not set in env, ask user to enable query_item or not. If the
    response is empty, use the default.
    Args:
      environ_cp: copy of the os.environ.
      var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
      query_item: string for feature related to the variable, e.g. "CUDA for
        Nvidia GPUs".
      enabled_by_default: boolean for default behavior.
      question: optional string for how to ask for user input.
      yes_reply: optional string for reply when feature is enabled.
      no_reply: optional string for reply when feature is disabled.
    Returns:
      boolean value of the variable.
    Raises:
      UserInputError: if an environment variable is set, but it cannot be
        interpreted as a boolean indicator, assume that the user has made a
        scripting error, and will continue to provide invalid input.
        Raise the error to avoid infinitely looping.
    """
    if not question:
        question = "Do you wish to build Larq Compute Engine with {} support?".format(
            query_item
        )
    if not yes_reply:
        yes_reply = "{} support will be enabled for Larq Compute Engine.".format(
            query_item
        )
    if not no_reply:
        no_reply = "No {}".format(yes_reply)

    yes_reply += "\n"
    no_reply += "\n"

    if enabled_by_default:
        question += " [Y/n]: "
    else:
        question += " [y/N]: "

    var = environ_cp.get(var_name)
    if var is not None:
        var_content = var.strip().lower()
        true_strings = ("1", "t", "true", "y", "yes")
        false_strings = ("0", "f", "false", "n", "no")
        if var_content in true_strings:
            var = True
        elif var_content in false_strings:
            var = False
        else:
            raise UserInputError(
                "Environment variable %s must be set as a boolean indicator.\n"
                "The following are accepted as TRUE : %s.\n"
                "The following are accepted as FALSE: %s.\n"
                "Current value is %s."
                % (var_name, ", ".join(true_strings), ", ".join(false_strings), var)
            )

    while var is None:
        user_input_origin = get_input(question)
        user_input = user_input_origin.strip().lower()
        if user_input == "y":
            print(yes_reply)
            var = True
        elif user_input == "n":
            print(no_reply)
            var = False
        elif not user_input:
            if enabled_by_default:
                print(yes_reply)
                var = True
            else:
                print(no_reply)
                var = False
        else:
            print("Invalid selection: {}".format(user_input_origin))
    return var


def run_shell(cmd, allow_non_zero=False, stderr=None):
    if stderr is None:
        stderr = sys.stdout
    if allow_non_zero:
        try:
            output = subprocess.check_output(cmd, stderr=stderr)
        except subprocess.CalledProcessError as e:
            output = e.output
    else:
        output = subprocess.check_output(cmd, stderr=stderr)
    return output.decode("UTF-8").strip()


def cygpath(path):
    """Convert path from posix to windows."""
    return os.path.abspath(path).replace("\\", "/")


def get_python_path(environ_cp, python_bin_path):
    """Get the python site package paths."""
    python_paths = []
    if environ_cp.get("PYTHONPATH"):
        python_paths = environ_cp.get("PYTHONPATH").split(":")
    try:
        stderr = open(os.devnull, "wb")
        library_paths = run_shell(
            [
                python_bin_path,
                "-c",
                'import site; print("\\n".join(site.getsitepackages()))',
            ],
            stderr=stderr,
        ).split("\n")
    except subprocess.CalledProcessError:
        library_paths = [
            run_shell(
                [
                    python_bin_path,
                    "-c",
                    "from distutils.sysconfig import get_python_lib;"
                    "print(get_python_lib())",
                ]
            )
        ]

    all_paths = set(python_paths + library_paths)

    paths = []
    for path in all_paths:
        if os.path.isdir(path):
            paths.append(path)
    return paths


def get_python_version(python_bin_path):
    """Get the python major version."""
    return run_shell([python_bin_path, "-c", "import sys; print(sys.version[:3])"])


def setup_python(environ_cp):
    """Setup python related env variables."""
    # Get PYTHON_BIN_PATH, default is the current running python.
    default_python_bin_path = sys.executable
    ask_python_bin_path = (
        "Please specify the location of python. [Default is " "{}]: "
    ).format(default_python_bin_path)
    while True:
        python_bin_path = get_from_env_or_user_or_default(
            environ_cp, "PYTHON_BIN_PATH", ask_python_bin_path, default_python_bin_path
        )
        # Check if the path is valid
        if os.path.isfile(python_bin_path) and os.access(python_bin_path, os.X_OK):
            break
        elif not os.path.exists(python_bin_path):
            print("Invalid python path: {} cannot be found.".format(python_bin_path))
        else:
            print(
                "{} is not executable.  Is it the python binary?".format(
                    python_bin_path
                )
            )
        environ_cp["PYTHON_BIN_PATH"] = ""

    # Convert python path to Windows style before checking lib and version
    if is_windows() or is_cygwin():
        python_bin_path = cygpath(python_bin_path)

    # Get PYTHON_LIB_PATH
    python_lib_path = environ_cp.get("PYTHON_LIB_PATH")
    if not python_lib_path:
        python_lib_paths = get_python_path(environ_cp, python_bin_path)
        if environ_cp.get("USE_DEFAULT_PYTHON_LIB_PATH") == "1":
            python_lib_path = python_lib_paths[0]
        else:
            print(
                "Found possible Python library paths:\n  %s"
                % "\n  ".join(python_lib_paths)
            )
            default_python_lib_path = python_lib_paths[0]
            python_lib_path = get_input(
                "Please input the desired Python library path to use. "
                "[Default is {}]\n".format(python_lib_paths[0])
            )
            if not python_lib_path:
                python_lib_path = default_python_lib_path
        environ_cp["PYTHON_LIB_PATH"] = python_lib_path

    python_version = get_python_version(python_bin_path)
    if python_version[0] == "2":
        write_to_bazelrc("build --host_force_python=PY2")
    print(
        f"Configuring builds with Python {python_version} support. To use a different "
        "Python version, re-run configuration inside a virtual environment or pass "
        "different binary/lib paths when prompted.\n"
    )

    # Convert python path to Windows style before writing into bazel.rc
    if is_windows() or is_cygwin():
        python_lib_path = cygpath(python_lib_path)

    # Set-up env variables used by python_configure.bzl
    write_action_env_to_bazelrc("PYTHON_BIN_PATH", python_bin_path)
    write_action_env_to_bazelrc("PYTHON_LIB_PATH", python_lib_path)
    write_to_bazelrc('build --python_path="{}"'.format(python_bin_path))
    environ_cp["PYTHON_BIN_PATH"] = python_bin_path

    # If choosen python_lib_path is from a path specified in the PYTHONPATH
    # variable, need to tell bazel to include PYTHONPATH
    if environ_cp.get("PYTHONPATH"):
        python_paths = environ_cp.get("PYTHONPATH").split(":")
        if python_lib_path in python_paths:
            write_action_env_to_bazelrc("PYTHONPATH", environ_cp.get("PYTHONPATH"))


def set_cc_opt_flags(environ_cp):
    """Set up architecture-dependent optimization flags.
    Also append CC optimization flags to bazel.rc.
    Args:
      environ_cp: copy of the os.environ.
    """
    if is_ppc64le():
        # gcc on ppc64le does not support -march, use mcpu instead
        default_cc_opt_flags = "-mcpu=native"
    elif is_windows():
        default_cc_opt_flags = "/arch:AVX"
    else:
        # On all other platforms, no longer use `-march=native` as this can result
        # in instructions that are too modern being generated. Users that want
        # maximum performance should compile TF in their environment and can pass
        # `-march=native` there.
        # See https://github.com/tensorflow/tensorflow/issues/45744 and duplicates
        default_cc_opt_flags = "-Wno-sign-compare"
    question = (
        "Please specify optimization flags to use during compilation when"
        ' bazel option "--config=opt" is specified [Default is %s]: '
    ) % default_cc_opt_flags
    cc_opt_flags = get_from_env_or_user_or_default(
        environ_cp, "CC_OPT_FLAGS", question, default_cc_opt_flags
    )
    for opt in cc_opt_flags.split():
        write_to_bazelrc("build:opt --copt=%s" % opt)
        write_to_bazelrc("build:opt --host_copt=%s" % opt)
    write_to_bazelrc("build:opt --define with_default_optimizations=true")


def maybe_set_manylinux_toolchain(environ_cp):
    write_to_bazelrc(
        "build:manylinux2010 --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.1:toolchain"
    )
    if get_var(
        environ_cp,
        var_name="MANYLINUX2010",
        query_item="manylinux2010-compatible pip package",
        enabled_by_default=False,
        question=(
            "Are you trying to build a manylinux2010-compatible pip package "
            "in the tensorflow:custom-op-ubuntu16 Docker container?"
        ),
        yes_reply="Building manylinux2010-compatible pip package.",
        no_reply="Not building manylinux2010-compatible pip package.",
    ):
        write_to_bazelrc("build --config=manylinux2010")
        write_to_bazelrc("test --config=manylinux2010")


def set_windows_build_flags(environ_cp):
    """Set Windows specific build options."""

    # First available in VS 16.4. Speeds up Windows compile times by a lot. See
    # https://groups.google.com/a/tensorflow.org/d/topic/build/SsW98Eo7l3o/discussion
    # pylint: disable=line-too-long
    write_to_bazelrc(
        "build --copt=/d2ReducedOptimizeHugeFunctions --host_copt=/d2ReducedOptimizeHugeFunctions"
    )

    if get_var(
        environ_cp,
        "TF_OVERRIDE_EIGEN_STRONG_INLINE",
        "Eigen strong inline",
        True,
        (
            "Would you like to override eigen strong inline for some C++ "
            "compilation to reduce the compilation time?"
        ),
        "Eigen strong inline overridden.",
        "Not overriding eigen strong inline, "
        "some compilations could take more than 20 mins.",
    ):
        # Due to a known MSVC compiler issue
        # https://github.com/tensorflow/tensorflow/issues/10521
        # Overriding eigen strong inline speeds up the compiling of
        # conv_grad_ops_3d.cc and conv_ops_3d.cc by 20 minutes,
        # but this also hurts the performance. Let users decide what they want.
        write_to_bazelrc("build --define=override_eigen_strong_inline=true")


def reset_lce_configure_bazelrc():
    """Reset file that contains customized config settings."""
    open(_LCE_BAZELRC, "w").close()


def main():
    # Make a copy of os.environ to be clear when functions and getting and setting
    # environment variables.
    environ_cp = dict(os.environ)

    reset_lce_configure_bazelrc()

    setup_python(environ_cp)

    set_cc_opt_flags(environ_cp)

    if is_windows():
        set_windows_build_flags(environ_cp)

    if is_linux():
        maybe_set_manylinux_toolchain(environ_cp)


if __name__ == "__main__":
    main()
