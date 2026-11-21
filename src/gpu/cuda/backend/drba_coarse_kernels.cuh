#pragma once

#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::drba_coarse_kernels {
namespace rn = kirari::gpu::cuda::rife_native;

inline constexpr int kConvTaps = 9;
inline constexpr int kChunk = 16;
inline constexpr int kWidth = 16;
inline constexpr int kHaloWidth = kWidth + 2;

template <int Channels>
inline constexpr int kPitch = Channels + 8;

__device__ __forceinline__ __half2 residual_pair(__half2 biased, __half2 old, float2 beta, float2 slope) {
  const auto v = __half22float2(biased);
  const __half2 scaled = __floats2half2_rn(v.x * beta.x, v.y * beta.y);
  const __half2 added = __hadd2(scaled, old);
  const auto f = rn::activate_pair(__half22float2(added), slope);
  return __floats2half2_rn(f.x, f.y);
}

template <int Channels, int Warps, int Rows>
constexpr int smem_bytes() {
  return (Warps * Rows + 2) * kHaloWidth * kPitch<Channels> * sizeof(__half);
}

template <int Channels, int Warps, int Rows>
__device__ __forceinline__ void load_halo(__half* tile, const rn::Params& p, int x0, int y0, int thread) {
  constexpr int halo_height = Warps * Rows + 2;
  constexpr int chunks = Channels / 8;
  constexpr int threads = Warps * 32;
  const auto base = rn::smem_u32(tile);
  for (int i = thread; i < halo_height * kHaloWidth * chunks; i += threads) {
    const int pixel = i / chunks, channel = (i % chunks) * 8;
    const int x = x0 + pixel % kHaloWidth - 1, y = y0 + pixel / kHaloWidth - 1;
    const bool valid = (x >= 0 && x < p.iw) && (y >= 0 && y < p.ih);
    const auto* source = valid ? p.in + (static_cast<long>(y) * p.iw + x) * p.in_ld + channel : p.in;
    rn::cp_async16(base + static_cast<std::uint32_t>((pixel * kPitch<Channels> + channel) * 2), source,
                   valid);
  }
  rn::cp_async_wait_all();
  __syncthreads();
}

template <int Channels, int CT, int Warps, int Rows, int TapUnroll = 1>
__global__ void __launch_bounds__(Warps * 32, 2)
    resident(rn::Params p, const float* scales, const float* slope) {
  static_assert(CT % 16 == 0 && Channels % CT == 0);
  constexpr int kInner = Channels / kChunk;
  constexpr int kNP = CT / 16;
  extern __shared__ __align__(16) unsigned char raw_smem[];
  auto* tile = reinterpret_cast<__half*>(raw_smem);
  const int thread = static_cast<int>(threadIdx.x), warp = thread / 32, lane = thread % 32;
  const int x0 = static_cast<int>(blockIdx.x) * kWidth, y0 = static_cast<int>(blockIdx.y) * (Warps * Rows);
  const int tile_id = static_cast<int>(blockIdx.z), channel0 = tile_id * CT;
  load_halo<Channels, Warps, Rows>(tile, p, x0, y0, thread);

  rn::Frag<rn::Acc::kF32> sums[Rows][CT / 8];
#pragma unroll
  for (int r = 0; r < Rows; ++r)
#pragma unroll
    for (int c = 0; c < CT / 8; ++c) sums[r][c].clear();

  const auto base = rn::smem_u32(tile) +
                    static_cast<std::uint32_t>(
                        (((warp * Rows * kHaloWidth + (lane & 15)) * kPitch<Channels>)+(lane >> 4) * 8) * 2);
  const auto* const weights = p.w + (tile_id * kConvTaps * kInner * kNP * 32) + lane;
  const auto fetch = [&](uint4(&values)[kNP], int q, int tap) {
#pragma unroll
    for (int n = 0; n < kNP; ++n) values[n] = __ldg(weights + ((tap * kInner + q) * kNP + n) * 32);
  };

  uint4 next[kNP];
  fetch(next, 0, 0);
  // Identique à conv_kernel<K16> pour chaque sortie : tranche K16, puis les neuf prises en ordre ligne.
#pragma unroll 1
  for (int q = 0; q < kInner; ++q) {
#pragma unroll TapUnroll
    for (int tap = 0; tap < kConvTaps; ++tap) {
      const int dy = tap / 3, dx = tap % 3;
      uint4 current[kNP];
#pragma unroll
      for (int n = 0; n < kNP; ++n) current[n] = next[n];
      if (tap + 1 < kConvTaps)
        fetch(next, q, tap + 1);
      else if (q + 1 < kInner)
        fetch(next, q + 1, 0);

#pragma unroll
      for (int row = 0; row < Rows; ++row) {
        std::uint32_t a[4];
        const auto offset =
            static_cast<std::uint32_t>((((row + dy) * kHaloWidth + dx) * kPitch<Channels> + q * kChunk) * 2);
        rn::ldmatrix4(a, base + offset);
#pragma unroll
        for (int n = 0; n < kNP; ++n) {
          sums[row][2 * n].mma(a, current[n].x, current[n].y);
          sums[row][2 * n + 1].mma(a, current[n].z, current[n].w);
        }
      }
    }
  }

  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int n = 0; n < CT / 8; ++n) {
    const int c = channel0 + n * 8 + t * 2;
    const auto bias = __ldg(reinterpret_cast<const float2*>(p.bias + c));
    const auto beta = __ldg(reinterpret_cast<const float2*>(scales + c));
    const auto relu = __ldg(reinterpret_cast<const float2*>(slope + c));
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
      const int local_y = warp * Rows + row;
      const int y = y0 + local_y;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int local_x = g + h * 8;
        const int x = x0 + local_x;
        if (x >= p.ow || y >= p.oh) continue;
        const auto skip = *reinterpret_cast<const __half2*>(
            tile + ((local_y + 1) * kHaloWidth + local_x + 1) * kPitch<Channels> + c);
        const auto sum = sums[row][n].pair(h);
        const auto biased = rn::to_half2(sum.x + bias.x, sum.y + bias.y);
        auto* out = reinterpret_cast<__half2*>(p.out + (static_cast<long>(y) * p.ow + x) * p.out_ld + c);
        *out = residual_pair(biased, skip, beta, relu);
      }
    }
  }
}

template <int Channels, int CT, int Warps, int Rows, int TapUnroll = 1>
void launch(rn::Params p, const float* beta, const float* slope, cudaStream_t stream) {
  constexpr int height = Warps * Rows;
  const dim3 grid((p.ow + kWidth - 1) / kWidth, (p.oh + height - 1) / height, Channels / CT);
  resident<Channels, CT, Warps, Rows, TapUnroll>
      <<<grid, Warps * 32, smem_bytes<Channels, Warps, Rows>(), stream>>>(p, beta, slope);
}

inline constexpr int kKc = 16, kCt = 32, kRows = 1, kPhaseRows = 2;
inline constexpr rn::Acc kAcc = rn::Acc::kF32;

// On charge huit demi-flottants contigus après le halo gauche de la tuile.
template <int Width, int Height, int Channels, int Rows, int Columns>
__device__ __forceinline__ void load_entry_slice(__half* tile, const __half* input, int thread, int ty,
                                                 int tx, int q) {
  constexpr int kStride = 2, kLds = kKc + 8, kGroup = 8;
  constexpr int kVectors = (Columns - 1) / kGroup;
  constexpr int kInputPlane = Width * Height;
  static_assert((Columns - 1) % kGroup == 0 && Width % kGroup == 0);
  // Après le halo gauche, les colonnes partent d'une adresse alignée.
  for (int i = thread; i < kKc * Rows * kVectors; i += rn::kThreads) {
    const int c = i / (Rows * kVectors);
    const int row = (i / kVectors) % Rows;
    const int column = i % kVectors * kGroup;
    const int y = ty * kStride - 1 + row, x = tx * kStride + column;
    const int channel = q * kKc + c;
    const bool valid = channel < Channels && y >= 0 && y < Height && x < Width;
    const auto* from = valid ? input + channel * kInputPlane + y * Width + x : input;
    const uint4 v = valid ? __ldg(reinterpret_cast<const uint4*>(from)) : make_uint4(0, 0, 0, 0);
    const std::uint32_t words[4]{v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      const int col = rn::smem_col<kStride, Columns>(column + j + 1);
      const auto bits = static_cast<std::uint16_t>(words[j / 2] >> (16 * (j & 1)));
      tile[(row * Columns + col) * kLds + c] = __ushort_as_half(bits);
    }
  }

  for (int i = thread; i < kKc * Rows; i += rn::kThreads) {
    const int row = i / kKc, c = i % kKc;
    const int y = ty * kStride - 1 + row, x = tx * kStride - 1;
    const int channel = q * kKc + c;
    const bool valid = channel < Channels && y >= 0 && y < Height && x >= 0;
    tile[row * Columns * kLds + c] =
        valid ? __ldg(input + channel * kInputPlane + y * Width + x) : __float2half(0.F);
  }
}

// L'entrée planaire charge huit demi-flottants à la fois avant le produit.
template <int kInputWidth, int kInputHeight, int kInputChannels, int kEntryPitch, bool Warm = false>
__global__ void __launch_bounds__(rn::kThreads, 2) entry_grouped(rn::Params p, float4* cache = nullptr) {
  constexpr int kStride = 2, kKernel = 3;
  using G = rn::Geometry<kKernel, kStride, kRows>;

  extern __shared__ __align__(16) unsigned char raw[];
  auto* tile = reinterpret_cast<__half*>(raw);
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / 32, lane = thread % 32;
  const int ty = static_cast<int>(blockIdx.y) * G::kTH;
  const int tx = static_cast<int>(blockIdx.x) * rn::kTW;
  const int z = static_cast<int>(blockIdx.z);
  const auto* weights = p.w + z * 9 * (kEntryPitch / kKc) * (kCt / 16) * 32;

  constexpr bool kCache = kInputChannels == 58;
  constexpr int kFixedChunks = 3;
  constexpr int kFragments = kRows * 2 * (kCt / 8);
  static_assert(!Warm || kCache);
  float4* partial = nullptr;
  if constexpr (kCache) {
    const auto id = (static_cast<std::size_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    partial = cache + kFragments * id * rn::kThreads + thread;
  }
  rn::Frag<kAcc> sums[kRows * 2][kCt / 8];
#pragma unroll
  for (int f = 0; f < kRows * 2; ++f)
#pragma unroll
    for (int n = 0; n < kCt / 8; ++n) {
      auto& sums4 = sums[f][n].r;
      if constexpr (Warm) {
        const auto v = __ldg(partial + (f * (kCt / 8) + n) * rn::kThreads);
        sums4[0] = v.x;
        sums4[1] = v.y;
        sums4[2] = v.z;
        sums4[3] = v.w;
      } else {
        sums[f][n].clear();
      }
    }
  constexpr int kStart = Warm ? kFixedChunks : 0;
#pragma unroll 1
  for (int q = kStart; q < kEntryPitch / kKc; ++q) {
    if (q != kStart) __syncthreads();
    load_entry_slice<kInputWidth, kInputHeight, kInputChannels, G::kIR, G::kIC>(tile, p.in, thread, ty, tx,
                                                                                q);

    __syncthreads();
    rn::accumulate<kKc, kCt, kKernel, kStride, kRows, kAcc>(sums, tile, weights, q, kEntryPitch / kKc, 0, 9,
                                                            warp, lane);
    if constexpr (kCache && !Warm) {
      // Les prises de la tranche 48–63 gardent les encodages restants et le temps.
      if (q + 1 == kFixedChunks) {
#pragma unroll
        for (int f = 0; f < kRows * 2; ++f)
#pragma unroll
          for (int n = 0; n < kCt / 8; ++n) {
            const auto& a = sums[f][n].r;
            partial[(f * (kCt / 8) + n) * rn::kThreads] = make_float4(a[0], a[1], a[2], a[3]);
          }
      }
    }
  }
  __syncthreads();
  rn::stage<kCt, kRows, kAcc>(sums, tile, p.bias + z * kCt, warp, lane);
  __syncthreads();
  rn::store<kCt, kRows, rn::Out::kNhwc>(tile, p, ty, tx, 0, z * kCt);
}

// Les quatre phases de la ConvTranspose gardent le rangement planaire.
__global__ void __launch_bounds__(rn::kThreads, 2) project_planar(rn::Params p) {
  constexpr int kHeight = rn::kWarps * kPhaseRows;
  extern __shared__ __align__(16) unsigned char raw[];
  auto* const tile = reinterpret_cast<__half*>(raw);
  const int t = static_cast<int>(threadIdx.x);
  const int warp = t / 32, lane = t % 32;
  const int ty = static_cast<int>(blockIdx.y) * kHeight;
  const int tx = static_cast<int>(blockIdx.x) * rn::kTW;
  const int z = static_cast<int>(blockIdx.z);
  const int phase = z / p.tiles, n0 = z % p.tiles * kCt;
  rn::Frag<kAcc> sums[kPhaseRows * 2][kCt / 8];
  rn::conv_sums<kKc, kCt, 2, 1, kPhaseRows, rn::In::kNhwc, kAcc, false>(sums, tile, p, ty, tx, z, 0, warp,
                                                                        lane);
  __syncthreads();
  rn::stage<kCt, kPhaseRows, kAcc>(sums, tile, p.bias + z * kCt, warp, lane);
  __syncthreads();
  for (int i = t; i < kCt * kHeight * rn::kTW; i += rn::kThreads) {
    const int c = i / (kHeight * rn::kTW), q = i % (kHeight * rn::kTW);
    const int y = ty + q / rn::kTW, x = tx + q % rn::kTW;
    if (y >= p.oh || x >= p.ow || n0 + c >= p.out_ch) continue;
    const int pixel = (2 * y + (phase >> 1)) * (2 * p.ow) + 2 * x + (phase & 1);
    p.out[(n0 + c) * (4 * p.oh * p.ow) + pixel] = tile[q * (kCt + 8) + c];
  }
}

}  // namespace kirari::gpu::cuda::drba_coarse_kernels
