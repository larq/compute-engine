#ifndef COMPUTE_ENGINE_KERNELS_PACKBITS_H_
#define COMPUTE_ENGINE_KERNELS_PACKBITS_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

#include "larq_compute_engine/core/types.h"
#ifdef __aarch64__
#include "larq_compute_engine/core/packbits_aarch64.h"
#endif

// From @flatbuffers, used for the FLATBUFFERS_LITTLEENDIAN macro
#include "flatbuffers/base.h"

namespace compute_engine {
namespace core {

// Utility functions
constexpr int GetPackedSize(int unpacked_elements, int bitwidth) {
  return (unpacked_elements + bitwidth - 1) / bitwidth;
}

constexpr int GetPackedMatrixSize(int rows, int cols, int bitwidth) {
  return rows * GetPackedSize(cols, bitwidth);
}

template <typename TBitpacked>
constexpr int GetPackedSize(int unpacked_elements) {
  return GetPackedSize(
      unpacked_elements,
      std::numeric_limits<
          typename std::make_unsigned<TBitpacked>::type>::digits);
}

template <typename TBitpacked>
constexpr int GetPackedMatrixSize(int rows, int cols) {
  return GetPackedMatrixSize(
      rows, cols,
      std::numeric_limits<
          typename std::make_unsigned<TBitpacked>::type>::digits);
}

template <class TIn, class TOut>
inline void pack(const TIn* fptr, TOut* buf) {
  constexpr std::size_t bitwidth = std::numeric_limits<TOut>::digits;
  *buf = 0;
  for (size_t i = 0; i < bitwidth; ++i) {
    if (fptr[i] < 0) *buf |= (TOut(1) << i);
  }
}

template <class TIn, class TOut>
inline void pack_quantized(const TIn* in, TOut* out,
                           const std::int32_t zero_point) {
  constexpr std::size_t bitwidth = std::numeric_limits<TOut>::digits;
  *out = 0;
  for (size_t i = 0; i < bitwidth; ++i) {
    // Note: uint8 to int32 will set the top 24 bits to 0
    //        int8 to int32 will set the top 24 bits to the int8 sign bit
    if (static_cast<std::int32_t>(in[i]) < zero_point) *out |= (TOut(1) << i);
  }
}

template <class T>
inline void pack(const T* fptr, std::uint8_t* buf) {
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

template <class T>
inline void pack(const T* fptr, std::uint32_t* buf) {
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

template <class T>
inline void pack(const T* fptr, std::uint64_t* buf) {
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

// Helper function
template <class TIn, class TOut>
inline void pack_bitfield(const TIn* in, TOut* out,
                          const std::int32_t zero_point) {
  // Note: The expressions in these if-statements are known at compile-time so
  // they are all optimied away
  constexpr bool is_quantized = !std::is_floating_point<TIn>::value;
  if (is_quantized)
    pack_quantized(in, out, zero_point);
  else
    pack(in, out);
}

template <class TIn, class TOut>
inline void packbits_array(const TIn* input_array, const std::size_t n,
                           TOut* bitpacked_array,
                           const std::int32_t zero_point) {
  constexpr std::size_t bitwidth = std::numeric_limits<TOut>::digits;

  int num_packed_elems = n / bitwidth;
  int elements_left = n - bitwidth * num_packed_elems;

  const TIn* in = input_array;
  TOut* out = bitpacked_array;

#ifdef __aarch64__
  if (FLATBUFFERS_LITTLEENDIAN && std::is_same<TIn, float>::value &&
      std::is_same<TOut, std::uint64_t>::value && zero_point == 0) {
    packbits_aarch64_64(reinterpret_cast<const float*>(in), num_packed_elems,
                        reinterpret_cast<std::uint64_t*>(out));
    in += bitwidth * num_packed_elems;
    out += num_packed_elems;
  } else {
    while (num_packed_elems--) {
      pack_bitfield(in, out++, zero_point);
      in += bitwidth;
    }
  }
#else
  while (num_packed_elems--) {
    pack_bitfield(in, out++, zero_point);
    in += bitwidth;
  }
#endif

  // If padding is needed, copy the remaining elements to a buffer and add
  // enough zeros to fill the bitwidth. This function assumes enough memory for
  // padding is already allocated in the output array `bitpacked_array`.
  if (elements_left != 0) {
    std::array<TIn, bitwidth> padding_buffer = {0};
    memcpy(padding_buffer.data(), in, elements_left * sizeof(TIn));
    for (size_t i = elements_left; i < bitwidth; ++i)
      padding_buffer[i] = zero_point;
    pack_bitfield(padding_buffer.data(), out, zero_point);
  }
}

// Bitpacks each row of a row-major matrix
template <class TIn, class TOut_>
inline void packbits_matrix(const TIn* input, const std::size_t input_num_rows,
                            const std::size_t input_num_cols, TOut_* output,
                            const std::int32_t zero_point = 0) {
  // Force the types to be unsigned so that the function can be called on signed
  // types as well
  using TOut = typename std::make_unsigned<TOut_>::type;

  constexpr std::size_t bitwidth = std::numeric_limits<TOut>::digits;

  const TIn* input_ptr = input;
  TOut* output_ptr = reinterpret_cast<TOut*>(output);

  if (input_num_cols % bitwidth == 0) {
    // If each row can be bitpacked without any padding, then we can treat
    // the matrix as one flat array and bitpack it all in one go.
    packbits_array(input_ptr, input_num_cols * input_num_rows, output_ptr,
                   zero_point);
  } else {
    // Calculate the size of the bitpacked rows
    const std::size_t output_num_cols = GetPackedSize<TOut>(input_num_cols);

    // Iterate through each row of the input matrix and bitpack the row into the
    // corresponding memory location of the output matrix
    for (size_t row_index = 0; row_index < input_num_rows; ++row_index) {
      packbits_array(input_ptr, input_num_cols, output_ptr, zero_point);
      input_ptr += input_num_cols;
      output_ptr += output_num_cols;
    }
  }
}

template <typename TBitpacked, typename TUnpacked>
void unpack_bitfield(const TBitpacked in, TUnpacked*& out,
                     std::size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    *out++ = (in & (1ULL << i)) ? TUnpacked(-1) : TUnpacked(1);
  }
}

// The matrix is stored in row-major mode.
// Every row is bitpacked as TBitpacked, with canonical order
// The argument `num_cols` is the *unpacked* number of cols!
// Enough output memory is assumed.
template <typename TBitpacked, typename TUnpacked>
inline void unpack_matrix(const TBitpacked* input_data,
                          const std::size_t num_rows,
                          const std::size_t num_cols, TUnpacked* output_data) {
  constexpr std::size_t bitwidth = std::numeric_limits<
      typename std::make_unsigned<TBitpacked>::type>::digits;

  const TBitpacked* in_ptr = input_data;
  TUnpacked* out_ptr = output_data;
  for (size_t row_index = 0; row_index < num_rows; ++row_index) {
    int num_full_blocks = num_cols / bitwidth;
    int elements_left = num_cols - bitwidth * num_full_blocks;

    while (num_full_blocks--) {
      unpack_bitfield(*in_ptr++, out_ptr, bitwidth);
    }

    if (elements_left != 0) {
      unpack_bitfield(*in_ptr++, out_ptr, elements_left);
    }
  }
}

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_PACKBITS_H_
