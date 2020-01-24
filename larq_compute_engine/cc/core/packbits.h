#ifndef COMPUTE_ENGINE_KERNELS_PACKBITS_H_
#define COMPUTE_ENGINE_KERNELS_PACKBITS_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "larq_compute_engine/cc/utils/types.h"
#ifdef __aarch64__
#include "larq_compute_engine/cc/core/packbits_aarch64.h"
#endif

namespace compute_engine {
namespace core {

#define GET_POINTER_TO_ROW(array_pointer, lda, row_index) \
  ((array_pointer) + ((row_index) * (lda)))

#define GET_ELEM_INDEX(row_index, col_index, lda) \
  ((row_index) * (lda) + (col_index))

template <class TIn, class TOut,
          bool UseZeroPoint = !std::is_floating_point<TIn>::value>
struct pack_bitfield_impl {};

template <class TIn, class TOut>
void pack_bitfield(const TIn* in, TOut* buf,
                   const std::int32_t zero_point = 0) {
  pack_bitfield_impl<TIn, TOut>::Run(in, buf, zero_point);
}

// Float floating-point datatypes without zero_point
template <class TIn, class TOut>
struct pack_bitfield_impl<TIn, TOut, false> {
  static void Run(const TIn* fptr, TOut* buf, const std::int32_t) {
    const size_t bitwidth = std::numeric_limits<TOut>::digits;
    *buf = 0;
    for (size_t i = 0; i < bitwidth; ++i) {
      if (fptr[i] < 0) *buf |= (TOut(1) << i);
    }
  }
};

// For fixed-point datatypes with a zero_point
template <class TIn, class TOut>
struct pack_bitfield_impl<TIn, TOut, true> {
  static void Run(const TIn* in, TOut* buf, const std::int32_t zero_point) {
    static_assert(std::is_same<TIn, std::uint8_t>::value ||
                      std::is_same<TIn, std::int8_t>::value,
                  "Quantized bitpack datatype must be uint8 or int8");
    const size_t bitwidth = std::numeric_limits<TOut>::digits;
    *buf = 0;
    for (size_t i = 0; i < bitwidth; ++i) {
      // Note: uint8 to int32 will set the top 24 bits to 0
      //        int8 to int32 will set the top 24 bits to the int8 sign bit
      if (static_cast<std::int32_t>(in[i]) < zero_point) *buf |= (TOut(1) << i);
    }
  }
};

template <>
struct pack_bitfield_impl<float, std::uint8_t, false> {
  static void Run(const float* fptr, std::uint8_t* buf, const std::int32_t) {
    struct bf {
      unsigned int b0 : 1;
      unsigned int b1 : 1;
      unsigned int b2 : 1;
      unsigned int b3 : 1;
      unsigned int b4 : 1;
      unsigned int b5 : 1;
      unsigned int b6 : 1;
      unsigned int b7 : 1;
    };

    union bf_u8 {
      bf t;
      std::uint8_t u8;
    };

    // TODO: use the bit sign instead of comparision operator
    // static_cast<const float>(1 << sizeof(a)*8-1);
    bf_u8 u;
    u.t.b0 = fptr[0] < 0;
    u.t.b1 = fptr[1] < 0;
    u.t.b2 = fptr[2] < 0;
    u.t.b3 = fptr[3] < 0;
    u.t.b4 = fptr[4] < 0;
    u.t.b5 = fptr[5] < 0;
    u.t.b6 = fptr[6] < 0;
    u.t.b7 = fptr[7] < 0;
    *buf = u.u8;
  }
};

template <>
struct pack_bitfield_impl<float, std::uint32_t, false> {
  static void Run(const float* fptr, std::uint32_t* buf, const std::int32_t) {
    struct bf {
      unsigned int b0 : 1;
      unsigned int b1 : 1;
      unsigned int b2 : 1;
      unsigned int b3 : 1;
      unsigned int b4 : 1;
      unsigned int b5 : 1;
      unsigned int b6 : 1;
      unsigned int b7 : 1;
      unsigned int b8 : 1;
      unsigned int b9 : 1;
      unsigned int b10 : 1;
      unsigned int b11 : 1;
      unsigned int b12 : 1;
      unsigned int b13 : 1;
      unsigned int b14 : 1;
      unsigned int b15 : 1;
      unsigned int b16 : 1;
      unsigned int b17 : 1;
      unsigned int b18 : 1;
      unsigned int b19 : 1;
      unsigned int b20 : 1;
      unsigned int b21 : 1;
      unsigned int b22 : 1;
      unsigned int b23 : 1;
      unsigned int b24 : 1;
      unsigned int b25 : 1;
      unsigned int b26 : 1;
      unsigned int b27 : 1;
      unsigned int b28 : 1;
      unsigned int b29 : 1;
      unsigned int b30 : 1;
      unsigned int b31 : 1;
    };

    union bf_u32 {
      bf t;
      std::uint32_t u32;
    };

    // TODO: use the bit sign instead of comparision operator
    // static_cast<const float>(1 << sizeof(a)*8-1);
    bf_u32 u;
    u.t.b0 = fptr[0] < 0;
    u.t.b1 = fptr[1] < 0;
    u.t.b2 = fptr[2] < 0;
    u.t.b3 = fptr[3] < 0;
    u.t.b4 = fptr[4] < 0;
    u.t.b5 = fptr[5] < 0;
    u.t.b6 = fptr[6] < 0;
    u.t.b7 = fptr[7] < 0;
    u.t.b8 = fptr[8] < 0;
    u.t.b9 = fptr[9] < 0;
    u.t.b10 = fptr[10] < 0;
    u.t.b11 = fptr[11] < 0;
    u.t.b12 = fptr[12] < 0;
    u.t.b13 = fptr[13] < 0;
    u.t.b14 = fptr[14] < 0;
    u.t.b15 = fptr[15] < 0;
    u.t.b16 = fptr[16] < 0;
    u.t.b17 = fptr[17] < 0;
    u.t.b18 = fptr[18] < 0;
    u.t.b19 = fptr[19] < 0;
    u.t.b20 = fptr[20] < 0;
    u.t.b21 = fptr[21] < 0;
    u.t.b22 = fptr[22] < 0;
    u.t.b23 = fptr[23] < 0;
    u.t.b24 = fptr[24] < 0;
    u.t.b25 = fptr[25] < 0;
    u.t.b26 = fptr[26] < 0;
    u.t.b27 = fptr[27] < 0;
    u.t.b28 = fptr[28] < 0;
    u.t.b29 = fptr[29] < 0;
    u.t.b30 = fptr[30] < 0;
    u.t.b31 = fptr[31] < 0;
    *buf = u.u32;
  }
};

#ifdef __aarch64__

// Template specialization for the float -> uint64 case
template <>
struct pack_bitfield_impl<float, std::uint64_t, false> {
  static void Run(const float* in, std::uint64_t* out, const std::int32_t) {
    return packbits_aarch64_64(in, out);
  }
};

#else

template <>
struct pack_bitfield_impl<float, std::uint64_t, false> {
  static void Run(const float* fptr, std::uint64_t* buf, const std::int32_t) {
    struct bf {
      unsigned int b0 : 1;
      unsigned int b1 : 1;
      unsigned int b2 : 1;
      unsigned int b3 : 1;
      unsigned int b4 : 1;
      unsigned int b5 : 1;
      unsigned int b6 : 1;
      unsigned int b7 : 1;
      unsigned int b8 : 1;
      unsigned int b9 : 1;
      unsigned int b10 : 1;
      unsigned int b11 : 1;
      unsigned int b12 : 1;
      unsigned int b13 : 1;
      unsigned int b14 : 1;
      unsigned int b15 : 1;
      unsigned int b16 : 1;
      unsigned int b17 : 1;
      unsigned int b18 : 1;
      unsigned int b19 : 1;
      unsigned int b20 : 1;
      unsigned int b21 : 1;
      unsigned int b22 : 1;
      unsigned int b23 : 1;
      unsigned int b24 : 1;
      unsigned int b25 : 1;
      unsigned int b26 : 1;
      unsigned int b27 : 1;
      unsigned int b28 : 1;
      unsigned int b29 : 1;
      unsigned int b30 : 1;
      unsigned int b31 : 1;
      unsigned int b32 : 1;
      unsigned int b33 : 1;
      unsigned int b34 : 1;
      unsigned int b35 : 1;
      unsigned int b36 : 1;
      unsigned int b37 : 1;
      unsigned int b38 : 1;
      unsigned int b39 : 1;
      unsigned int b40 : 1;
      unsigned int b41 : 1;
      unsigned int b42 : 1;
      unsigned int b43 : 1;
      unsigned int b44 : 1;
      unsigned int b45 : 1;
      unsigned int b46 : 1;
      unsigned int b47 : 1;
      unsigned int b48 : 1;
      unsigned int b49 : 1;
      unsigned int b50 : 1;
      unsigned int b51 : 1;
      unsigned int b52 : 1;
      unsigned int b53 : 1;
      unsigned int b54 : 1;
      unsigned int b55 : 1;
      unsigned int b56 : 1;
      unsigned int b57 : 1;
      unsigned int b58 : 1;
      unsigned int b59 : 1;
      unsigned int b60 : 1;
      unsigned int b61 : 1;
      unsigned int b62 : 1;
      unsigned int b63 : 1;
    };

    union bf_u64 {
      bf t;
      std::uint64_t u64;
    };

    bf_u64 u;
    u.t.b0 = fptr[0] < 0;
    u.t.b1 = fptr[1] < 0;
    u.t.b2 = fptr[2] < 0;
    u.t.b3 = fptr[3] < 0;
    u.t.b4 = fptr[4] < 0;
    u.t.b5 = fptr[5] < 0;
    u.t.b6 = fptr[6] < 0;
    u.t.b7 = fptr[7] < 0;
    u.t.b8 = fptr[8] < 0;
    u.t.b9 = fptr[9] < 0;
    u.t.b10 = fptr[10] < 0;
    u.t.b11 = fptr[11] < 0;
    u.t.b12 = fptr[12] < 0;
    u.t.b13 = fptr[13] < 0;
    u.t.b14 = fptr[14] < 0;
    u.t.b15 = fptr[15] < 0;
    u.t.b16 = fptr[16] < 0;
    u.t.b17 = fptr[17] < 0;
    u.t.b18 = fptr[18] < 0;
    u.t.b19 = fptr[19] < 0;
    u.t.b20 = fptr[20] < 0;
    u.t.b21 = fptr[21] < 0;
    u.t.b22 = fptr[22] < 0;
    u.t.b23 = fptr[23] < 0;
    u.t.b24 = fptr[24] < 0;
    u.t.b25 = fptr[25] < 0;
    u.t.b26 = fptr[26] < 0;
    u.t.b27 = fptr[27] < 0;
    u.t.b28 = fptr[28] < 0;
    u.t.b29 = fptr[29] < 0;
    u.t.b30 = fptr[30] < 0;
    u.t.b31 = fptr[31] < 0;
    u.t.b32 = fptr[32] < 0;
    u.t.b33 = fptr[33] < 0;
    u.t.b34 = fptr[34] < 0;
    u.t.b35 = fptr[35] < 0;
    u.t.b36 = fptr[36] < 0;
    u.t.b37 = fptr[37] < 0;
    u.t.b38 = fptr[38] < 0;
    u.t.b39 = fptr[39] < 0;
    u.t.b40 = fptr[40] < 0;
    u.t.b41 = fptr[41] < 0;
    u.t.b42 = fptr[42] < 0;
    u.t.b43 = fptr[43] < 0;
    u.t.b44 = fptr[44] < 0;
    u.t.b45 = fptr[45] < 0;
    u.t.b46 = fptr[46] < 0;
    u.t.b47 = fptr[47] < 0;
    u.t.b48 = fptr[48] < 0;
    u.t.b49 = fptr[49] < 0;
    u.t.b50 = fptr[50] < 0;
    u.t.b51 = fptr[51] < 0;
    u.t.b52 = fptr[52] < 0;
    u.t.b53 = fptr[53] < 0;
    u.t.b54 = fptr[54] < 0;
    u.t.b55 = fptr[55] < 0;
    u.t.b56 = fptr[56] < 0;
    u.t.b57 = fptr[57] < 0;
    u.t.b58 = fptr[58] < 0;
    u.t.b59 = fptr[59] < 0;
    u.t.b60 = fptr[60] < 0;
    u.t.b61 = fptr[61] < 0;
    u.t.b62 = fptr[62] < 0;
    u.t.b63 = fptr[63] < 0;
    *buf = u.u64;
  }
};

#endif

template <class TIn, class TOut>
inline void packbits_array(const TIn* input_array, const std::size_t n,
                           TOut* bitpacked_array,
                           const std::int32_t zero_point = 0) {
  constexpr size_t bitwidth = std::numeric_limits<TOut>::digits;

  int num_packed_elems = n / bitwidth;
  int elements_left = n - bitwidth * num_packed_elems;

  const TIn* in = input_array;
  TOut* out = bitpacked_array;

  while (num_packed_elems--) {
    pack_bitfield(in, out++, zero_point);
    in += bitwidth;
  }

  // If padding is needed, copy the remaining elements to a buffer and add
  // enough zeros to fill the bitwidth. This function assumes enough memory for
  // padding is already allocatd in the output array `bitpacked_array`.
  if (elements_left != 0) {
    std::array<TIn, bitwidth> padding_buffer = {0};
    memcpy(padding_buffer.data(), in, elements_left * sizeof(TIn));
    if (!std::is_same<TIn, float>::value) {
      for (size_t i = elements_left; i < bitwidth; ++i)
        padding_buffer[i] = static_cast<TIn>(zero_point);
    }
    pack_bitfield(padding_buffer.data(), out, zero_point);
  }
}

// input/output matrices are stored in row-major mode
// bitpacking_axis argument specifies the dimension in matrix where
// the bitpacking operation is performed. For example for a RowWise
// bitpacking operation compresses an MxN matrix to a Mx(N/bitwidth)
// matrix.
//
// zero_point is only for quantized types
template <class TIn, class TOutContainer>
inline void packbits_matrix(const TIn* input_data, const size_t input_num_rows,
                            const size_t input_num_cols, TOutContainer& output,
                            size_t& output_num_rows, size_t& output_num_cols,
                            size_t& output_bitpadding,
                            const Axis bitpacking_axis,
                            const std::int32_t zero_point = 0) {
  using TOut = typename TOutContainer::value_type;
  const size_t bitwidth = std::numeric_limits<TOut>::digits;

  if (bitpacking_axis == Axis::RowWise) {
    // calculate size of bitpacked matrix and allocate its memory
    output_num_rows = input_num_rows;
    output_num_cols = (input_num_cols + bitwidth - 1) / bitwidth;

    const auto output_size = output_num_cols * output_num_rows;
    output.resize(output_size);

    const auto input_row_size = input_num_cols;
    const auto output_row_size = output_num_cols;

    const auto num_extra_elems = input_num_cols % bitwidth;
    output_bitpadding = num_extra_elems == 0 ? 0 : bitwidth - num_extra_elems;

    // iterate through each row of the input matrix and bitpack the row into
    // the corresponding memory location of the output matrix
    auto output_data = output.data();
    for (size_t row_index = 0; row_index < input_num_rows; ++row_index)
      packbits_array(
          GET_POINTER_TO_ROW(input_data, input_row_size, row_index),
          input_row_size,
          GET_POINTER_TO_ROW(output_data, output_row_size, row_index),
          zero_point);
    return;
  }

  if (bitpacking_axis == Axis::ColWise) {
    // calculate size of bitpacked matrix and allocate its memory
    output_num_rows = (input_num_rows + bitwidth - 1) / bitwidth;
    output_num_cols = input_num_cols;
    const auto output_size = output_num_cols * output_num_rows;
    output.resize(output_size);

    const auto input_row_size = input_num_cols;
    const auto output_row_size = output_num_cols;

    const auto num_extra_elems = input_num_rows % bitwidth;
    output_bitpadding = num_extra_elems == 0 ? 0 : bitwidth - num_extra_elems;

    // allocate temporary buffers
    std::vector<TIn> input_buffer(input_num_rows);
    std::vector<TOut> output_buffer(output_num_rows);

    auto output_data = output.data();

    // iterate through the columns
    for (size_t col_index = 0; col_index < input_num_cols; ++col_index) {
      // store the values of the current column in a buffer
      for (size_t row_index = 0; row_index < input_num_rows; ++row_index)
        input_buffer[row_index] =
            input_data[GET_ELEM_INDEX(row_index, col_index, input_row_size)];

      // bitpack the buffer and store it in the the output matrix
      packbits_array(input_buffer.data(), input_buffer.size(),
                     output_buffer.data(), zero_point);

      // store the bitpacked values of the current column in the output matrix
      for (size_t row_index = 0; row_index < output_num_rows; ++row_index)
        output_data[GET_ELEM_INDEX(row_index, col_index, output_row_size)] =
            output_buffer[row_index];
    }
    return;
  }
}

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_PACKBITS_H_
