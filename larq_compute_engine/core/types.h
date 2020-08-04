#ifndef COMPUTE_ENGINE_CORE_TYPES_H_
#define COMPUTE_ENGINE_CORE_TYPES_H_

namespace compute_engine {
namespace core {

// defines the memory layout of the filter values
enum class FilterFormat { Unknown, HWIO, OHWI, OHWI_PACKED };

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_TYPES_H_
