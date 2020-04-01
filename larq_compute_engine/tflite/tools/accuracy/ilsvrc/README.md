## Accuracy evaluation for ILSVRC 2012 (Imagenet Large Scale Visual Recognition Challenge) image classification task

This binary can evaluate the accuracy of TFLite models trained for the [ILSVRC 2012 image classification task]
(http://www.image-net.org/challenges/LSVRC/2012/).
The binary takes the path to validation images and labels as inputs. It outputs the accuracy after running the TFLite model on the validation sets.

## Parameters
The binary takes the following parameters:

*   `model_file` : `string` \
    Path to the TFlite model file.

*   `ground_truth_images_path`: `string` \
    The path to the directory containing ground truth images.

*   `ground_truth_labels`: `string` \
    Path to ground truth labels file. This file should contain the same number
    of labels as the number images in the ground truth directory. The labels are
    assumed to be in the same order as the sorted filename of images. For your
    convenience, the ImageNet validation
    [ground_truth_labels.txt](ground_truth_labels.txt) is included in this
    directory.

*   `model_output_labels`: `string` \
    Path to the file containing labels, that is used to interpret the output of
    the model. For most models trained on ImageNet using TFDS, you can use
    [model_output_labels.txt](model_output_labels.txt) in this directory.

*   `output_file_path`: `string` \
    This is the path to the output file. The output is a CSV file that has
    top-10 accuracies in each row. Each line of output file is the cumulative
    accuracy after processing images in a sorted order. So first line is
    accuracy after processing the first image, second line is accuracy after
    processing first two images. The last line of the file is accuracy after
    processing the entire validation set.

and the following optional parameters:

*   `blacklist_file_path`: `string` \
    Path to blacklist file. This file contains the indices of images that are
    blacklisted for evaluation. 1762 images are blacklisted in ILSVRC dataset.
    For details please refer to readme.txt of ILSVRC2014 devkit.

*   `num_images`: `int` (default=0) \
    The number of images to process, if 0, all images in the directory are
    processed otherwise only num_images will be processed.

*   `num_threads`: `int` (default=4) \
    The number of threads to use for evaluation. Note: This does not change the
    number of TFLite Interpreter threads, but shards the dataset to speed up
    evaluation.

*   `proto_output_file_path`: `string` \
    Optionally, the computed accuracies can be output to a file as a
    string-serialized instance of tflite::evaluation::TopkAccuracyEvalMetrics.

*   `num_ranks`: `int` (default=10) \
    The number of top-K accuracies to return. For example, if num_ranks=5, top-1
    to top-5 accuracy fractions are returned.

The following optional parameters can be used to modify the inference runtime:

*   `num_interpreter_threads`: `int` (default=1) \
    This modifies the number of threads used by the TFLite Interpreter for
    inference.

*   `delegate`: `string` \
    If provided, tries to use the specified delegate for accuracy evaluation.
    Valid values: "nnapi", "gpu".

## Running the binary

### Preparation

(1) Download and extract the ImageNet validation images, as JPEGs, into a
    folder.

(2) Download the ground truth labels `.txt` file:

```bash
wget https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image/imagenet2012_validation_labels.txt -O ground_truth_labels.txt
```

(3) Download the model output labels `.txt` file:

```bash
wget https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image/imagenet2012_labels.txt -O model_output_labels.txt
```

### On Android

(0) Refer to https://docs.larq.dev/compute-engine/quickstart_android/ for configuring NDK and SDK.

(1) Build using the following command:

```
bazel build -c opt \
  --config=android_arm64 \
  //larq_compute_engine/tflite/tools/accuracy/ilsvrc:imagenet_accuracy_eval
```

(2) Connect your phone. Push the binary to your phone with adb push
     (make the directory if required):

```
adb push imagenet_accuracy_eval /data/local/tmp
```

(3) Make the binary executable.

```
adb shell chmod +x /data/local/tmp/imagenet_accuracy_eval
```

(4) Push the TFLite model  that you need to test. For example:

```
adb push quicknet.tflite /data/local/tmp
```

(5) Push the imagenet images to device, make sure device has sufficient storage available before pushing the dataset:

```
adb shell mkdir /data/local/tmp/ilsvrc_images && \
adb push ${IMAGES_DIR} /data/local/tmp/ilsvrc_images
```

(6) Push the generated validation ground labels to device.

```
adb push ${GROUND_TRUTH_LABELS_TXT} /data/local/tmp/ground_truth_labels.txt
```

(7) Push the model labels text file to device.

```
adb push ${MODEL_LABELS_TXT} /data/local/tmp/model_output_labels.txt
```

(8) Run the binary.

```
adb shell /data/local/tmp/imagenet_accuracy_eval \
  --model_file=/data/local/tmp/quicknet.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ground_truth_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images.
```

###  On Desktop

(1) Build and run using the following command:

```
bazel run -c opt \
  //tflite/tools/accuracy/ilsvrc:imagenet_accuracy_eval \
  --model_file=quicknet.tflite \
  --ground_truth_images_path=${IMAGES_DIR} \
  --ground_truth_labels=${GROUND_TRUTH_LABELS_TXT} \
  --model_output_labels=${MODEL_LABELS_TXT} \
  --output_file_path=/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images.
```
