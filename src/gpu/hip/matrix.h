#pragma once

#include <hip/hip_fp16.h>

#if defined(__HIP_PLATFORM_NVIDIA__)
#include <mma.h>
#endif

namespace kirari::gpu::hip_backend::matrix {
struct Samples {
  float values[8];
};

#if defined(__HIP_PLATFORM_AMD__)
using Half = _Float16;
using Fragment = Half __attribute__((ext_vector_type(16)));
using A = Fragment;
using B = Fragment;
using Accumulator = Fragment;

__device__ Fragment load(const Half* base, int stride, int column) {
  Fragment out;
  __builtin_memcpy(&out, __builtin_assume_aligned(base + column * stride, 16), sizeof(out));
  return out;
}

__device__ Accumulator zero() {
  return {};
}

__device__ Accumulator add(A a, B b, Accumulator c) {
  return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c, false);
}

__device__ Samples elements(Accumulator acc, int, int) {
  Samples result{};
#pragma unroll
  for (int i = 0; i < 8; ++i) result.values[i] = static_cast<float>(acc[2 * i]);
  return result;
}

__device__ A left(const Half* base, int stride, int column) {
  return load(base, stride, column);
}
__device__ B right(const Half* base, int stride, int column) {
  return load(base, stride, column);
}

__device__ A point_left(const Half* src, int base, int pixels, int stride, int k, int column) {
  if (base + column >= pixels) return {};
  return load(src + base * stride + k, stride, column);
}
#else
using Half = __half;
namespace wm = nvcuda::wmma;
using A = wm::fragment<wm::matrix_a, 16, 16, 16, Half, wm::row_major>;
using B = wm::fragment<wm::matrix_b, 16, 16, 16, Half, wm::col_major>;
using Accumulator = wm::fragment<wm::accumulator, 16, 16, 16, Half>;

__device__ Accumulator zero() {
  Accumulator c;
  wm::fill_fragment(c, __float2half(0.F));
  return c;
}

__device__ A staged_left(const Half* base, int stride, int column, bool valid) {
  // Certaines lignes du halo décalent A de 16 octets. WMMA exige une adresse alignée à 32.
  // Le même rangement remplit les bords incomplets de la projection par zéro.
  __shared__ __align__(32) Half rows[4][16 * 16];
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  Half* tile = rows[wave];
  const int k = lane / 16 * 8;
  const uint4 packed = valid ? *reinterpret_cast<const uint4*>(base + column * stride + k) : uint4{};
  *reinterpret_cast<uint4*>(tile + column * 16 + k) = packed;
  __syncwarp();
  A a;
  wm::load_matrix_sync(a, tile, 16);
  __syncwarp();
  return a;
}

__device__ A left(const Half* base, int stride, int column) {
  return staged_left(base, stride, column, true);
}

__device__ A point_left(const Half* src, int base, int pixels, int stride, int k, int column) {
  const Half* origin = base < pixels ? src + base * stride + k : src;
  return staged_left(origin, stride, column, base + column < pixels);
}

__device__ B right(const Half* base, int stride, int) {
  B b;
  wm::load_matrix_sync(b, base, stride);
  return b;
}

__device__ Accumulator add(A a, B b, Accumulator c) {
  wm::mma_sync(c, a, b, c);
  return c;
}

__device__ Samples elements(Accumulator acc, int wave, int lane) {
  __shared__ __align__(32) Half values[128 / 32][16 * 16];
  Half* tile = values[wave];
  wm::store_matrix_sync(tile, acc, 16, wm::mem_row_major);
  __syncwarp();
  Samples result{};
#pragma unroll
  for (int i = 0; i < 8; ++i) result.values[i] = __half2float(tile[(2 * i + lane / 16) * 16 + lane % 16]);
  __syncwarp();
  return result;
}
#endif

}  // namespace kirari::gpu::hip_backend::matrix
