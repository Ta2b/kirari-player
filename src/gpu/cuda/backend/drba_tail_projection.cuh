#pragma once

#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::drba_tail_projection {
namespace rn = rife_native;
using Sum = rn::Frag<rn::Acc::kF32>;
constexpr int kChannels = 96, kChunk = 16, kChunks = kChannels / kChunk;
constexpr int kOutputPitch = 24, kBiasPitch = 32, kSharedPitch = kChunk + 8;
constexpr int kWidth = 16, kRows = 2;

namespace {

// Un CTA projette deux parités horizontales et partage leur prise centrale.
struct Tile {
  static constexpr int height = kRows * rn::kWarps;
  static constexpr int input_height = height + 1;
  static constexpr int input_width = kWidth + 2;
  static constexpr int bytes = input_height * input_width * kSharedPitch * sizeof(__half);
};

struct Weight {
  uint4 lo;
  uint2 hi;
};

__device__ __forceinline__ Weight load_weight(const uint4* packed, int tap, int chunk) {
  const auto* v = packed + (tap * kChunks + chunk) * 64;
  return {__ldg(v), __ldg(reinterpret_cast<const uint2*>(v + 32))};
}

__device__ __forceinline__ void dot(Sum (&sum)[3], const std::uint32_t (&values)[4], const Weight& w) {
  sum[0].mma(values, w.lo.x, w.lo.y);
  sum[1].mma(values, w.lo.z, w.lo.w);
  sum[2].mma(values, w.hi.x, w.hi.y);
}

__device__ __forceinline__ void emit(const Sum (&sum)[2][kRows][3], const rn::Params& p, int tx, int ty,
                                     int py, int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int f = 0; f < kRows; ++f) {
    const int y = ty + warp * kRows + f, origin = tx + g;
    if (y >= p.oh) continue;
#pragma unroll
    for (int n = 0; n < 3; ++n) {
      const int channel = n * 8 + 2 * t;
      const auto even_bias = __ldg(reinterpret_cast<const float2*>(p.bias + (2 * py) * kBiasPitch + channel));
      const auto odd_bias =
          __ldg(reinterpret_cast<const float2*>(p.bias + (2 * py + 1) * kBiasPitch + channel));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int x = origin + 8 * h;
        if (x >= p.ow) continue;
        const auto out = (static_cast<long>(2 * y + py) * (2 * p.ow) + 2 * x) * kOutputPitch + channel;
        const auto even = sum[0][f][n].pair(h);
        const auto odd = sum[1][f][n].pair(h);
        *reinterpret_cast<__half2*>(p.out + out) = rn::to_half2(even.x + even_bias.x, even.y + even_bias.y);
        *reinterpret_cast<__half2*>(p.out + out + kOutputPitch) =
            rn::to_half2(odd.x + odd_bias.x, odd.y + odd_bias.y);
      }
    }
  }
}

// Les poids restent au pas CT32. Les trois groupes utiles calculent les vingt
// plans et les quatre zéros de remplissage. Chaque phase garde l'ordre K16,
// puis prise, du noyau transposé de référence.
__global__ void __launch_bounds__(rn::kThreads, 2) project(rn::Params p) {
  extern __shared__ __align__(16) unsigned char raw[];
  auto* tile = reinterpret_cast<__half*>(raw);
  const int tx = static_cast<int>(blockIdx.x) * kWidth, ty = static_cast<int>(blockIdx.y) * Tile::height;
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int py = static_cast<int>(blockIdx.z);
  constexpr int phase_stride = 4 * kChunks * 64;
  const auto* even = p.w + 2 * py * phase_stride + lane;
  const auto* odd = even + phase_stride;
  const auto base =
      rn::smem_u32(tile) + static_cast<std::uint32_t>(((lane & 15) * kSharedPitch + (lane >> 4) * 8) * 2);
  Sum sum[2][kRows][3];
#pragma unroll
  for (auto& phase : sum)
#pragma unroll
    for (auto& fragment : phase)
#pragma unroll
      for (auto& value : fragment) value.clear();

#pragma unroll 1
  for (int q = 0; q < kChunks; ++q) {
    if (q != 0) __syncthreads();
    rn::load_nhwc<kChunk, 1, Tile::input_height, Tile::input_width, false>(tile, p, q * kChunk, ty + py - 1,
                                                                           tx - 1);
    rn::cp_async_wait_all();
    __syncthreads();
#pragma unroll 1
    for (int dy = 0; dy < 2; ++dy) {
      const int tap = dy * 2;
      const auto e0 = load_weight(even, tap, q);
      const auto e1 = load_weight(even, tap + 1, q);
      const auto o0 = load_weight(odd, tap, q);
      const auto o1 = load_weight(odd, tap + 1, q);
#pragma unroll
      for (int f = 0; f < kRows; ++f) {
        const int y = warp * kRows + f + dy;
        const auto addr = base + static_cast<std::uint32_t>(y * Tile::input_width * kSharedPitch * 2);
        std::uint32_t middle[4], side[4];
        rn::ldmatrix4(side, addr);
        dot(sum[0][f], side, e0);
        rn::ldmatrix4(middle, addr + kSharedPitch * 2);
        dot(sum[0][f], middle, e1);
        dot(sum[1][f], middle, o0);
        rn::ldmatrix4(side, addr + 2 * kSharedPitch * 2);
        dot(sum[1][f], side, o1);
      }
    }
  }
  emit(sum, p, tx, ty, py, warp, lane);
}

}  // namespace

inline void launch(const rn::Params& p, cudaStream_t stream) {
  const dim3 grid((p.ow + kWidth - 1) / kWidth, (p.oh + Tile::height - 1) / Tile::height, 2);
  project<<<grid, rn::kThreads, Tile::bytes, stream>>>(p);
}
}  // namespace kirari::gpu::cuda::drba_tail_projection
