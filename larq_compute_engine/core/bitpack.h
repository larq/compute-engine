#ifndef COMPUTE_ENGINE_KERNELS_BITPACK_H_
#define COMPUTE_ENGINE_KERNELS_BITPACK_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

#include "larq_compute_engine/core/types.h"
#ifdef __aarch64__
#include "larq_compute_engine/core/bitpack_aarch64.h"
#endif

// From @flatbuffers, used for the FLATBUFFERS_LITTLEENDIAN macro
#include "flatbuffers/base.h"

namespace compute_engine {
namespace core {

// Utility functions

constexpr int GetPackedSize(int unpacked_elements) {
  return (unpacked_elements + bitpacking_bitwidth - 1) / bitpacking_bitwidth;
}

constexpr int GetPackedMatrixSize(int rows, int cols) {
  return rows * GetPackedSize(cols);
}

template <class TIn>
inline void pack_quantized(const TIn* in, TBitpacked* out,
                           const std::int32_t zero_point) {
  *out = 0;
  for (size_t i = 0; i < bitpacking_bitwidth; ++i) {
    // Note: uint8 to int32 will set the top 24 bits to 0
    //        int8 to int32 will set the top 24 bits to the int8 sign bit
    if (static_cast<std::int32_t>(in[i]) < zero_point)
      *out |= (TBitpacked(1) << i);
  }
}

template <class T>
inline void pack(const T* fptr, TBitpacked* buf) {
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

  union bf_i32 {
    bf t;
    TBitpacked i32;
  };

  // TODO: use the bit sign instead of comparision operator
  // static_cast<const float>(1 << sizeof(a)*8-1);
  bf_i32 i;
  i.t.b0 = fptr[0] < 0;
  i.t.b1 = fptr[1] < 0;
  i.t.b2 = fptr[2] < 0;
  i.t.b3 = fptr[3] < 0;
  i.t.b4 = fptr[4] < 0;
  i.t.b5 = fptr[5] < 0;
  i.t.b6 = fptr[6] < 0;
  i.t.b7 = fptr[7] < 0;
  i.t.b8 = fptr[8] < 0;
  i.t.b9 = fptr[9] < 0;
  i.t.b10 = fptr[10] < 0;
  i.t.b11 = fptr[11] < 0;
  i.t.b12 = fptr[12] < 0;
  i.t.b13 = fptr[13] < 0;
  i.t.b14 = fptr[14] < 0;
  i.t.b15 = fptr[15] < 0;
  i.t.b16 = fptr[16] < 0;
  i.t.b17 = fptr[17] < 0;
  i.t.b18 = fptr[18] < 0;
  i.t.b19 = fptr[19] < 0;
  i.t.b20 = fptr[20] < 0;
  i.t.b21 = fptr[21] < 0;
  i.t.b22 = fptr[22] < 0;
  i.t.b23 = fptr[23] < 0;
  i.t.b24 = fptr[24] < 0;
  i.t.b25 = fptr[25] < 0;
  i.t.b26 = fptr[26] < 0;
  i.t.b27 = fptr[27] < 0;
  i.t.b28 = fptr[28] < 0;
  i.t.b29 = fptr[29] < 0;
  i.t.b30 = fptr[30] < 0;
  i.t.b31 = fptr[31] < 0;
  *buf = i.i32;
}

// Helper function
template <class TIn>
inline void pack_bitfield(const TIn* in, TBitpacked* out,
                          const std::int32_t zero_point) {
  // Note: The expressions in these if-statements are known at compile-time so
  // they are all optimied away
  constexpr bool is_quantized = !std::is_floating_point<TIn>::value;
  if (is_quantized)
    pack_quantized(in, out, zero_point);
  else
    pack(in, out);
}

template <class TIn>
inline void bitpack_array(const TIn* input_array, const std::size_t n,
                          TBitpacked* bitpacked_array,
                          const std::int32_t zero_point) {
  int num_packed_elems = n / bitpacking_bitwidth;
  int elements_left = n - bitpacking_bitwidth * num_packed_elems;

  const TIn* in = input_array;
  TBitpacked* out = bitpacked_array;

#ifdef __aarch64__
  if (FLATBUFFERS_LITTLEENDIAN && std::is_same<TIn, float>::value &&
      zero_point == 0) {
    const int num_4x32_blocks = num_packed_elems / 4;
    bitpack_aarch64_4x32(reinterpret_cast<const float*>(in), num_4x32_blocks,
                         out);
    in += bitpacking_bitwidth * 4 * num_4x32_blocks;
    out += 4 * num_4x32_blocks;
    num_packed_elems %= 4;
  }
#endif
  while (num_packed_elems--) {
    pack_bitfield(in, out++, zero_point);
    in += bitpacking_bitwidth;
  }

  // If padding is needed, copy the remaining elements to a buffer and add
  // enough zeros to fill the bitpacking_bitwidth. This function assumes enough
  // memory for padding is already allocated in the output array
  // `bitpacked_array`.
  if (elements_left != 0) {
    std::array<TIn, bitpacking_bitwidth> padding_buffer = {{0}};
    memcpy(padding_buffer.data(), in, elements_left * sizeof(TIn));
    for (size_t i = elements_left; i < bitpacking_bitwidth; ++i)
      padding_buffer[i] = zero_point;
    pack_bitfield(padding_buffer.data(), out, zero_point);
  }
}

// Bitpacks each row of a row-major matrix
template <class TIn>
inline void bitpack_matrix(const TIn* input, const std::size_t input_num_rows,
                           const std::size_t input_num_cols, TBitpacked* output,
                           const std::int32_t zero_point = 0) {
  if (input_num_cols % bitpacking_bitwidth == 0) {
    // If each row can be bitpacked without any padding, then we can treat
    // the matrix as one flat array and bitpack it all in one go.
    bitpack_array(input, input_num_cols * input_num_rows, output, zero_point);
  } else {
    // Calculate the size of the bitpacked rows
    const std::size_t output_num_cols = GetPackedSize(input_num_cols);

    // Iterate through each row of the input matrix and bitpack the row into the
    // corresponding memory location of the output matrix
    for (size_t row_index = 0; row_index < input_num_rows; ++row_index) {
      bitpack_array(input, input_num_cols, output, zero_point);
      input += input_num_cols;
      output += output_num_cols;
    }
  }
}

template <typename TUnpacked>
void unpack_bitfield(const TBitpacked in, TUnpacked*& out,
                     std::size_t num_elements, const TUnpacked zero_bit_result,
                     const TUnpacked one_bit_result) {
  for (size_t i = 0; i < num_elements; ++i) {
    *out++ = (in & (TBitpacked(1) << i)) ? one_bit_result : zero_bit_result;
  }
}

// The matrix is stored in row-major mode.
// Every row is bitpacked as TBitpacked, with canonical order
// The argument `num_cols` is the *unpacked* number of cols!
// Enough output memory is assumed.
template <typename TUnpacked>
inline void unpack_matrix(const TBitpacked* input_data,
                          const std::size_t num_rows,
                          const std::size_t num_cols, TUnpacked* output_data,
                          const TUnpacked zero_bit_result = TUnpacked(+1),
                          const TUnpacked one_bit_result = TUnpacked(-1)) {
  const TBitpacked* in_ptr = input_data;
  TUnpacked* out_ptr = output_data;
  for (size_t row_index = 0; row_index < num_rows; ++row_index) {
    int num_full_blocks = num_cols / bitpacking_bitwidth;
    int elements_left = num_cols - bitpacking_bitwidth * num_full_blocks;

    while (num_full_blocks--) {
      unpack_bitfield(*in_ptr++, out_ptr, bitpacking_bitwidth, zero_bit_result,
                      one_bit_result);
    }

    if (elements_left != 0) {
      unpack_bitfield(*in_ptr++, out_ptr, elements_left, zero_bit_result,
                      one_bit_result);
    }
  }
}

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_BITPACK_H_