#!/usr/bin/env bash
set -e -x

# Remove the now private ppa. This can be removed after the docker image removes the
# pre-installed python packages from this ppa.
rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list

ln -sf /usr/bin/python3.5 /usr/bin/python3 # Py36 has issues with add-apt
add-apt-repository -y ppa:deadsnakes/ppa

apt-get -y -qq update

apt-get -y -qq install python${PYTHON_VERSION}-dev --no-install-recommends
ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python
ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

ls /usr/include/x86_64-linux-gnu/
ls /dt7/usr/include/x86_64-linux-gnu/

# This really shouldn't be required to get 3.7 builds working
ln -sf /usr/include/x86_64-linux-gnu/python3.7m /dt7/usr/include/x86_64-linux-gnu/python3.7m || true
ln -sf /usr/include/x86_64-linux-gnu/python3.7m /dt8/usr/include/x86_64-linux-gnu/python3.7m || true

curl https://bootstrap.pypa.io/get-pip.py | python
python --version
python -m pip --version

curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/bazelisk-linux-amd64 > /usr/bin/bazelisk
chmod +x /usr/bin/bazelisk

python -m pip install numpy six --no-cache-dir

yes | ./configure.sh

# Build
bazelisk build :build_pip_pkg --copt=-fvisibility=hidden

# Package Whl
bazel-bin/build_pip_pkg artifacts
