/**
 * Adapted from https://github.com/finnickniu/tensorflow_object_detection_tflite/blob/master/demo.cpp
 * Additional reference from https://github.com/aiden-dai/ai-tflite-opencv/blob/master/object_detection/test_camera.py
 * Last reference for normalizing input from https://stackoverflow.com/questions/42266742/how-to-normalize-image-in-opencv
 * A simple demo object detection application which loads model.tflite and labelmap.txt from 
 * current directory and performs inference on camera output
 **/
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cmath>
#include <chrono>

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <opencv2/opencv.hpp>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// change these variables accordingly
#define INPUT_SIZE 224
#define MODEL_FILE "model.tflite"
#define LABEL_MAP_FILE "labelmap.txt"
#define NUM_THREADS 4
#define DO_INFERENCE 1
#define MAX_BOXES 50

#define NO_PRESS 255

using namespace tflite;
using namespace std;
using namespace cv;
using namespace std::chrono;

struct Object{
    cv::Rect rec;
    int      class_id;
    float    prob;
};

std::vector<std::string> initialize_labels() {
    // get the list of labels from labelmap
    std::vector<std::string> labels;
    std::ifstream input( LABEL_MAP_FILE );
    for( std::string line; getline( input, line ); )
    {
        labels.push_back( line);
    }
    return labels;
}

void test() {
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(MODEL_FILE);

    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    compute_engine::tflite::RegisterLCECustomOps(&resolver);

    InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<Interpreter> interpreter;
    builder(&interpreter, NUM_THREADS);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    std::vector<std::string> labels = initialize_labels();

    std::cout << "Initialized interpreter and labels" << std::endl;

    // declare the camera
    auto cam = cv::VideoCapture();

    cam.open(1, cv::CAP_V4L);

    // get camera resolution
    auto cam_width = cam.get(cv::CAP_PROP_FRAME_WIDTH);
    auto cam_height = cam.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Got the camera, see cam_width and cam_height: " 
                << cam_width << ',' << cam_height << std::endl;

    // initialize the FPS tracking variables
    high_resolution_clock::time_point previous_frame_time = high_resolution_clock::now();
    high_resolution_clock::time_point current_frame_time;

    // allocate tensor before inference loop
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    // start camera loop
    while (true) {
        // declare image buffers
        cv::Mat original_image;
        cv::Mat resized_image;

        // declare the output buffers
        TfLiteTensor* output_boxes_tensor = nullptr;
        TfLiteTensor* output_scores_tensor = nullptr;
        TfLiteTensor* output_classes_tensor = nullptr;

        std::vector<float> locations_vector;
        std::vector<float> scores_vector;
        std::vector<int> classes_vector;

        // read frame from camera
        auto success = cam.read(original_image);
        if (!success) {
            std::cout << "cam fail" << std::endl;
            break;
        }

#if DO_INFERENCE == 1
        // Resize the original image
        resize(original_image, resized_image, Size(INPUT_SIZE, INPUT_SIZE));

        // Convert image color (assume image was BGR)
        cvtColor(resized_image, resized_image, COLOR_BGR2RGB);

        // Convert input image to Float and normalize
        resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255, 0);

        // Declare the input
        float* input = interpreter->typed_input_tensor<float>(0);

        // feed input
        memcpy(input, resized_image.data, resized_image.total() * resized_image.elemSize());

        // run inference
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

        // get outputs
        output_boxes_tensor = interpreter->output_tensor(0);
        auto tmp_output_boxes = output_boxes_tensor->data.f;

        output_scores_tensor = interpreter->output_tensor(1);
        auto tmp_output_scores = output_scores_tensor->data.f;

        output_classes_tensor   = interpreter->output_tensor(2);
        auto tmp_output_classes = output_classes_tensor->data.i32;

        // get number of detections in output
        auto num_detections = output_scores_tensor->bytes / sizeof(int32_t);

        for (int i = 0; i < num_detections; i++){
            for (int j = 0; j < 4; j++) {
                auto output_box_coord = tmp_output_boxes[i * 4 + j];
                locations_vector.push_back(output_box_coord);
            }
            scores_vector.push_back(tmp_output_scores[i]);
            classes_vector.push_back(tmp_output_classes[i]);
            // std::cout << tmp_output_classes[i] << std::endl;
        }

        // scale the output boxes, then add them into a vector of bounding box objects
        int count=0;
        std::vector<Object> objects;

        for (int j = 0; j <locations_vector.size(); j+=4) {
            auto xmin = locations_vector[j] * cam_width;
            auto ymin = locations_vector[j+1] * cam_height;
            auto xmax = locations_vector[j+2] * cam_width;
            auto ymax = locations_vector[j+3] * cam_height;

            auto width = xmax - xmin;
            auto height = ymax - ymin;
            
            Object object;
            object.class_id = classes_vector[count];
            object.rec.x = xmin;
            object.rec.y = ymin;
            object.rec.width = width;
            object.rec.height = height;
            object.prob = scores_vector[count];
            objects.push_back(object);

            count+=1;
        }

        // show the bounding boxes on GUI
        for (int l = 0; l < objects.size(); l++)
        {			
            Object object = objects.at(l);
            auto score = object.prob;
            auto score_rounded = ((float) ((int) (score * 100 + 0.5)) / 100);

            Scalar color = Scalar(255, 0, 0);
            auto class_id = object.class_id;
            auto class_label = labels[class_id];

            std::ostringstream label_txt_stream;
            label_txt_stream << class_label << " (" << score_rounded << ")";
            std::string label_txt = label_txt_stream.str();

            cv::rectangle(original_image, object.rec, color, 1);
            cv::putText(original_image, 
                        label_txt, 
                        cv::Point(object.rec.x, object.rec.y - 5),
                        cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));
        }
#endif

        // calculate FPS
        current_frame_time = high_resolution_clock::now();
        auto delta_time = 
            duration_cast<microseconds>(current_frame_time - previous_frame_time); 
        previous_frame_time = current_frame_time;
        auto fps = pow(10, 6) / delta_time.count();
        auto fps_string = format("%.2f FPS", fps);
        
        // put FPS on screen
        cv::putText(original_image, fps_string,
                    cv::Point(40, 40),
                    cv::FONT_HERSHEY_COMPLEX, 1.0, 
                    cv::Scalar(10, 255, 50), 
                    2);
        
        // show image on screen
        cv::imshow("cam", original_image);

        // go to next frame after 30ms if no key pressed
        auto k = cv::waitKey(30) & 0xFF;
        if (k != NO_PRESS) {
            std::cout << "See k: " << k << std::endl;
            break;
        }
    }
}

int main(int argc, char** argv) {
    test();
    return 0;
}