#ifndef COMPUTE_ENGINE_ARM32_BGEMM_H_
#define COMPUTE_ENGINE_ARM32_BGEMM_H_

namespace compute_engine {
namespace core {

// TODO: choose better namespace?
// TODO: choose better function signature compatible with RUY
void bgemm_arm32(const size_t m, const size_t n, const size_t k,
                 const uint64_t* a, const size_t lda, const uint64_t* b,
                 const size_t ldb, int32_t* c, const size_t ldc) {
  // TODO
  return;
}
}  // namespace core
}  // namespace compute_engine

#endif
