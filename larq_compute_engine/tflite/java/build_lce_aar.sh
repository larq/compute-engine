#!/bin/bash

# this is a bash script based on tensorflow lite android AAR build script

set -e
set -x

TMPDIR=`mktemp -d`
trap "rm -rf $TMPDIR" EXIT

VERSION=1.0

BUILDER=bazel
BASEDIR=larq_compute_engine/tflite
CROSSTOOL="//external:android/crosstool"
HOST_CROSSTOOL="@bazel_tools//tools/cpp:toolchain"

BUILD_OPTS="-c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a"
CROSSTOOL_OPTS="--crosstool_top=$CROSSTOOL --host_crosstool_top=$HOST_CROSSTOOL"

test -d $BASEDIR || (echo "Aborting: not at top-level build directory"; exit 1)

function build_lce_aar() {
  local OUTDIR=$1
  $BUILDER build $BUILD_OPTS $BASEDIR/java:tensorflow-lite-lce.aar
  unzip -d $OUTDIR $BUILDER-bin/$BASEDIR/java/tensorflow-lite-lce.aar
  # targetSdkVersion is here to prevent the app from requesting spurious
  # permissions, such as permission to make phone calls. It worked for v1.0,
  # but minSdkVersion might be the preferred way to handle this.
  sed -i -e 's/<application>/<uses-sdk android:targetSdkVersion="25"\/><application>/' $OUTDIR/AndroidManifest.xml

  $BUILDER build $BUILD_OPTS $BASEDIR/java:tensorflowlite_java
  # override the classes.jar with the Java sources from TF Lite Java API
  cp $BUILDER-bin/$BASEDIR/java/libtensorflowlite_java.jar $OUTDIR/classes.jar
}

# function build_arch() {
#   local ARCH=$1
#   local CONFIG=$2
#   local OUTDIR=$3
#   mkdir -p $OUTDIR/jni/$ARCH/
#   $BUILDER build $BUILD_OPTS $CROSSTOOL_OPTS --cpu=$CONFIG \
#     $BASEDIR/java:libtensorflowlite_jni.so
#   cp $BUILDER-bin/$BASEDIR/java/libtensorflowlite_jni.so $OUTDIR/jni/$ARCH/
# }

rm -rf $TMPDIR
mkdir -p $TMPDIR/jni

build_lce_aar $TMPDIR
# build_arch arm64-v8a arm64-v8a $TMPDIR
# build_arch armeabi-v7a armeabi-v7a $TMPDIR
# build_arch x86 x86 $TMPDIR
# build_arch x86_64 x86_64 $TMPDIR

AAR_FILE=`grealpath tflite-lce-${VERSION}.aar`
(cd $TMPDIR && zip $AAR_FILE -r *)
echo "New AAR file is $AAR_FILE"
