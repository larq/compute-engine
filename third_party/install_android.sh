#!/usr/bin/env bash
set -e

# **NOTE**: This requires Java 8 and won't work on never versions. See:
# https://stackoverflow.com/questions/46402772/failed-to-install-android-sdk-java-lang-noclassdeffounderror-javax-xml-bind-a

# Taken from tensorflow/lite/tools/tflite-android.Dockerfile

# default LCE Android Env. variables
export ANDROID_SDK_URL="https://dl.google.com/android/repository/commandlinetools-linux-6858069_latest.zip"
export ANDROID_HOME="/tmp/lce_android"
export ANDROID_API_LEVEL=30
export ANDROID_BUILD_TOOLS_VERSION=31.0.0
export ANDROID_NDK_VERSION=25.2.9519653
export ANDROID_NDK_API_LEVEL=30


# download android SDK
mkdir -p $ANDROID_HOME; cd $ANDROID_HOME;

echo -e "Downloading Android SDK ... "
curl -o lce_android_sdk.zip $ANDROID_SDK_URL;
echo -e "DONE.\n\n"

echo -e "Unpacking Android SDK ... "
unzip lce_android_sdk.zip -d /tmp
mkdir -p ${ANDROID_HOME}/cmdline-tools
mv /tmp/cmdline-tools ${ANDROID_HOME}/cmdline-tools/latest
echo -e "DONE.\n\n"

rm lce_android_sdk.zip;

# install android platform and build tools
echo -e "Updating SDK manager ... "
yes | $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --licenses
$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --update
echo -e "DONE.\n\n"

echo -e "Installing Android SDK Platform and Build Tools ... "
$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager \
    "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
    "platforms;android-${ANDROID_API_LEVEL}" \
    "platform-tools"
echo -e "DONE.\n\n"

echo -e "Installing Android NDK ... "
$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager \
    "ndk;${ANDROID_NDK_VERSION}"
echo -e "DONE.\n\n"
