#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace tensorflow {

using std::string;

pybind11::bytes ConvertGraphDefToTFLiteFlatBuffer(
    const pybind11::bytes& graphdef_bytes,
    const std::vector<string>& input_arrays,
    const std::vector<string>& input_dtypes,
    const std::vector<std::vector<int>>& input_shapes,
    const std::vector<string>& output_arrays, const bool should_quantize,
    const std::string& target_str, const pybind11::object& default_ranges,
    const bool experimental_enable_bitpacked_activations);

pybind11::bytes ConvertSavedModelToTFLiteFlatBuffer(
    const std::string& saved_model_dir,
    const std::vector<std::string>& saved_model_tags,
    const std::vector<std::string>& exported_names,
    const int saved_model_version, const std::string& target_str,
    const pybind11::object& default_ranges,
    const bool experimental_enable_bitpacked_activations);
}  // namespace tensorflow

PYBIND11_MODULE(_tf_tfl_flatbuffer, m) {
  m.def("convert_graphdef_to_tflite_flatbuffer",
        &tensorflow::ConvertGraphDefToTFLiteFlatBuffer);
  m.def("convert_saved_model_to_tflite_flatbuffer",
        &tensorflow::ConvertSavedModelToTFLiteFlatBuffer);
};
