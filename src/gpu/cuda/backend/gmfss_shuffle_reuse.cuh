#pragma once

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::gmfss_shuffle_reuse {
namespace rn = rife_native;
inline constexpr int kRingRows = 128;

template <int Rows, bool Ring = false>
constexpr int smem_bytes() {
  using G = rn::Geometry<3, 1, Rows>;
  constexpr int in = G::kIR * G::kIC * (64 + 8) * sizeof(__half);
  constexpr int out = G::kTH * rn::kTW * (64 + 8) * sizeof(__half);
  return in + (Ring ? 0 : out);
}

template <int Rows>
__device__ __forceinline__ void store_ring(const rn::Frag<rn::Acc::kF16> (&acc)[Rows * 2][8],
                                           const rn::Params& p, const float* phase_bias, int x0, int y0,
                                           int warp, int lane, int phase) {
  constexpr int kChannels = 64;
  const int pixel = lane / 4, pair = lane % 4;
  const bool odd = (pair & 1) != 0;
  const int slot = (pair >> 1) | ((pair & 1) << 1);
  float2 bias[8];
#pragma unroll
  for (int n = 0; n < 8; ++n) bias[n] = __ldg(reinterpret_cast<const float2*>(phase_bias + n * 8 + 2 * pair));
#pragma unroll
  for (int f = 0; f < Rows * 2; ++f) {
    const int y = y0 + warp * Rows + f / 2;
    if (y >= p.oh) continue;
    const long row = static_cast<long>((2 * y + (phase >> 1)) & (kRingRows - 1)) * (2 * p.ow);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int x = x0 + 16 * (f & 1) + pixel + 8 * h;
      const long offset = (row + 2L * x + (phase & 1)) * kChannels;
      // Quatre lanes traitent un pixel. L'échange entre voisines conserve les deux
      // mots FP16x2 et permet un store aligné de quatre canaux, sans tampon de sortie.
#pragma unroll
      for (int n = 0; n < 8; n += 2) {
        const auto packed = [&](int i) {
          const float2 v = acc[f][i].pair(h);
          const __half2_raw value = rn::to_half2(v.x + bias[i].x, v.y + bias[i].y);
          return std::uint32_t{value.x} | (std::uint32_t{value.y} << 16U);
        };
        const auto lo = packed(n), hi = packed(n + 1);
        const auto neighbor = __shfl_xor_sync(0xffffffffU, odd ? lo : hi, 1, 4);
        const uint2 words = odd ? make_uint2(neighbor, hi) : make_uint2(lo, neighbor);
        if (x < p.ow) *reinterpret_cast<uint2*>(p.out + offset + n * 8 + 4 * slot) = words;
      }
    }
  }
}

// PixelShuffle(2) : le halo et le PReLU éventuel servent à plusieurs phases.
// Les poids, biais, ordre des sommes et stores sont ceux des phases indépendantes.
template <int Rows, int Phases, rn::Acc A, bool Ring>
__device__ __forceinline__ void shuffle(rn::Params p, int first_y, __half* input) {
  static_assert(Phases == 2 || Phases == 4);
  static_assert(!Ring || A == rn::Acc::kF16);
  using G = rn::Geometry<3, 1, Rows>;
  constexpr int kKc = 64, kCt = 64, kSteps = kKc / 16;
  auto* output = input + G::kIR * G::kIC * (kKc + 8);
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int y0 = first_y + static_cast<int>(blockIdx.y) * G::kTH;
  const int x0 = static_cast<int>(blockIdx.x) * rn::kTW;
  const int first = static_cast<int>(blockIdx.z) * Phases;
  rn::load_nhwc<kKc, 1, G::kIR, G::kIC>(input, p, 0, y0 - 1, x0 - 1);
  rn::cp_async_wait_all();
  __syncthreads();
  if (p.in_slope != 1.F) {
    rn::prelu_tile<kKc, G::kIR * G::kIC>(input, p.in_slope);
    __syncthreads();
  }
#pragma unroll 1
  for (int phase = first; phase < first + Phases; ++phase) {
    rn::Frag<A> acc[Rows * 2][kCt / 8];
#pragma unroll
    for (int f = 0; f < Rows * 2; ++f)
#pragma unroll
      for (int nt = 0; nt < kCt / 8; ++nt) acc[f][nt].clear();
    const uint4* weights = p.w + phase * 9 * kSteps * (kCt / 16) * 32;
    rn::accumulate<kKc, kCt, 3, 1, Rows, A, false>(acc, input, weights, 0, kSteps, 0, 9, warp, lane);
    const float* bias = p.bias + phase * kCt;
    if constexpr (Ring) {
      store_ring<Rows>(acc, p, bias, x0, y0, warp, lane, phase);
    } else {
      rn::stage<kCt, Rows, A>(acc, output, bias, warp, lane);
      __syncthreads();
      rn::store<kCt, Rows, rn::Out::kPhase>(output, p, y0, x0, phase, 0);
      if (phase + 1 < first + Phases) __syncthreads();  // réemploi du tampon de sortie
    }
  }
}
template <int Rows, int Phases, rn::Acc A>
__global__ void __launch_bounds__(rn::kThreads, 2) conv_shuffle(rn::Params p) {
  extern __shared__ __align__(16) unsigned char shared[];
  shuffle<Rows, Phases, A, false>(p, 0, reinterpret_cast<__half*>(shared));
}

static __global__ void __launch_bounds__(rn::kThreads, 2) conv_shuffle_ring(rn::Params p, int first_y) {
  extern __shared__ __align__(16) unsigned char shared[];
  shuffle<1, 2, rn::Acc::kF16, true>(p, first_y, reinterpret_cast<__half*>(shared));
}
}  // namespace kirari::gpu::cuda::gmfss_shuffle_reuse
