FROM ubuntu:18.04

RUN apt-get update && apt-get install curl unzip build-essential openjdk-8-jdk python3-dev python3-pip qemu-user -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.4.0/bazelisk-linux-amd64 > /usr/local/bin/bazelisk && chmod +x /usr/local/bin/bazelisk
RUN ln -s /usr/bin/python3 /usr/local/bin/python && ln -s /usr/bin/pip3 /usr/local/bin/pip
RUN pip install six numpy --no-cache-dir

WORKDIR /compute-engine
COPY . .
RUN ./third_party/install_android.sh
RUN yes n | ./configure.sh
RUN bazelisk --version
