#!/usr/bin/env bash
set -e

# default LCE Android Env. variables
export ANDROID_SDK_URL="https://dl.google.com/android/repository/sdk-tools-linux-3859397.zip"
export ANDROID_HOME="/tmp/lce_android"
export ANDROID_VERSION=29
export ANDROID_BUILD_TOOLS_VERSION=28.0.3
export ANDROID_NDK_VERSION=19.2.5345600

# download android SDK
mkdir -p $ANDROID_HOME; cd $ANDROID_HOME;

echo -e "Downloading Android SDK ... "
curl -o lce_android_sdk.zip $ANDROID_SDK_URL;
echo -e "DONE.\n\n"

echo -e "Unpacking Android SDK ... "
unzip lce_android_sdk.zip;
echo -e "DONE.\n\n"

rm lce_android_sdk.zip;

# install android platform and build tools
echo -e "Updating SDK manager ... "
yes | $ANDROID_HOME/tools/bin/sdkmanager --licenses
$ANDROID_HOME/tools/bin/sdkmanager --update
echo -e "DONE.\n\n"

echo -e "Installing Android SDK Platform and Build Tools ... "
$ANDROID_HOME/tools/bin/sdkmanager "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
    "platforms;android-${ANDROID_VERSION}" \
    "platform-tools"
echo -e "DONE.\n\n"

echo -e "Installing Android NDK ... "
$ANDROID_HOME/tools/bin/sdkmanager \
    "ndk;${ANDROID_NDK_VERSION}"
echo -e "DONE.\n\n"
