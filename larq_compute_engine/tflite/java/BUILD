load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@org_tensorflow//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("@org_tensorflow//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# Building tensorflowlite_lce.aar including 4 variants of .so
# To build the LCE compatible TFLite AAR, run the following command:
# bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
#   larq_compute_engine/tflite/java:tensorflowlite_lce
aar_with_jni(
    name = "tensorflow-lite-lce",
    android_library = ":tensorflowlite_lce",
)

android_library(
    name = "tensorflowlite_lce",
    javacopts = JAVACOPTS,
    manifest = "@org_tensorflow//tensorflow/lite/java:AndroidManifest.xml",
    proguard_specs = ["@org_tensorflow//tensorflow/lite/java:proguard.flags"],
    deps = [
        ":tensorflowlite_lce_native",
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "tensorflowlite_java",
    resource_jars =[
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite",
    ],
)

cc_library(
    name = "tensorflowlite_lce_native",
    srcs = ["libtensorflowlite_jni.so"],
    visibility = ["//visibility:private"],
)

tflite_jni_binary(
    name = "libtensorflowlite_jni.so",
    deps = [
        "@org_tensorflow//tensorflow/lite/delegates/nnapi/java/src/main/native",
        "@org_tensorflow//tensorflow/lite/java/src/main/native",
    ],
)
