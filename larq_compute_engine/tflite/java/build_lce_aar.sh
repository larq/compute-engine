#!/bin/bash

# This is a bash script based on TFLite android AAR build script.
# Additionally, we extract the TFLite Java API sources from the
# `libtensorflowlite_java.jar` target and replace the `classes.jar` file of
# LCE AAR with them.

set -e
set -x

TMPDIR=`mktemp -d`
trap "rm -rf $TMPDIR" EXIT

VERSION=1.0

BUILDER="${BUILDER:-bazelisk}"
BASEDIR=larq_compute_engine/tflite
CROSSTOOL="//external:android/crosstool"
HOST_CROSSTOOL="@bazel_tools//tools/cpp:toolchain"

BUILD_OPTS="-c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a"
CROSSTOOL_OPTS="--crosstool_top=$CROSSTOOL --host_crosstool_top=$HOST_CROSSTOOL"

test -d $BASEDIR || (echo "Aborting: not at top-level build directory"; exit 1)

function build_lce_aar() {
  local OUTDIR=$1
  $BUILDER build $BUILD_OPTS $BASEDIR/java:tensorflow-lite-lce.aar
  unzip -d $OUTDIR bazel-bin/$BASEDIR/java/tensorflow-lite-lce.aar
  # targetSdkVersion is here to prevent the app from requesting spurious
  # permissions, such as permission to make phone calls. It worked for v1.0,
  # but minSdkVersion might be the preferred way to handle this.
  sed -i -e 's/<application>/<uses-sdk android:targetSdkVersion="25"\/><application>/' $OUTDIR/AndroidManifest.xml

  $BUILDER build $BUILD_OPTS $BASEDIR/java:tensorflowlite_java
  # override the classes.jar with the Java sources from TF Lite Java API
  cp bazel-bin/$BASEDIR/java/libtensorflowlite_java.jar $OUTDIR/classes.jar
}

rm -rf $TMPDIR
mkdir -p $TMPDIR/jni

build_lce_aar $TMPDIR

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    AAR_FILE=`realpath tflite-lce-${VERSION}.aar`
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # on macOS get 'grealpath' by installing 'coreutils' package:
    # "brew install coreutils"
    AAR_FILE=`grealpath tflite-lce-${VERSION}.aar`
else
    # Unknown.
    echo "ERROR: could not detect the OS."
    exit 1
fi

(cd $TMPDIR && zip $AAR_FILE -r *)
echo "New AAR file is $AAR_FILE"
