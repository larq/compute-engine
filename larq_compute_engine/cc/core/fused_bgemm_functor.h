#ifndef COMPUTE_ENGINE_KERNELS_FUSED_BGEMM_FUNCTORS_H_
#define COMPUTE_ENGINE_KERNELS_FUSED_BGEMM_FUNCTORS_H_

#include "larq_compute_engine/cc/core/bgemm_functor.h"
#include "larq_compute_engine/cc/core/packbits.h"
#include "larq_compute_engine/cc/utils/macros.h"

namespace compute_engine {
namespace core {

// this functor just encapsulates the bitpacking and bgemm in one function
template <class TIn1, class TIn2, class TOut, class TBitpacked,
          class TBGemmFunctor>
class FusedBGemmFunctor {
 public:
  void operator()(const size_t m, const size_t n, const size_t k,
                  const TIn1* a_data, const size_t lda, const TIn2* b_data,
                  const size_t ldb, TOut* c_data_out, const size_t ldc) {
    const int a_num_rows = m;
    const int a_num_cols = k;
    std::vector<TBitpacked> a_data_bp;
    size_t a_num_rows_bp = 0, a_num_cols_bp = 0;
    size_t a_bitpadding = 0;
    packbits_matrix(a_data, a_num_rows, a_num_cols, a_data_bp, a_num_rows_bp,
                    a_num_cols_bp, a_bitpadding, Axis::RowWise);

    const int b_num_rows = k;
    const int b_num_cols = n;
    std::vector<TBitpacked> b_data_bp;
    size_t b_num_rows_bp = 0, b_num_cols_bp = 0;
    size_t b_bitpadding = 0;
    packbits_matrix(b_data, b_num_rows, b_num_cols, b_data_bp, b_num_rows_bp,
                    b_num_cols_bp, b_bitpadding, Axis::ColWise);

    const int m_bp = a_num_rows_bp;
    const int k_bp = a_num_cols_bp;
    const int n_bp = b_num_cols_bp;
    const int lda_bp = k_bp;
    const int ldb_bp = n_bp;
    const int ldc_bp = n_bp;
    TBGemmFunctor bgemm_functor;
    bgemm_functor(m_bp, n_bp, k_bp, a_data_bp.data(), lda_bp, b_data_bp.data(),
                  ldb_bp, c_data_out, ldc_bp, a_bitpadding);
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_FUSED_BGEMM_FUNCTORS_H_
