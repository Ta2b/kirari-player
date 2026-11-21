// La colle X2 fournit les quatre tranches mobiles à la convolution du dernier bloc.
#pragma once

#include <type_traits>

#include "gpu/cuda/backend/drba_fused_kernels.h"
#include "gpu/cuda/backend/drba_fused_sampling.cuh"
#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::drba_hot {
namespace {

namespace rn = rife_native;
using drba_sample::before_side;
using drba_sample::enlarge;
using drba_sample::sample;
using drba_sample::SourceZ;
using drba_sample::Tap;
using drba_sample::tap_of;
constexpr int kKc = 16, kCt = 48, kRows = 1, kS = 2, kKh = 3;
using G = rn::Geometry<kKh, kS, kRows>;
constexpr int kCells = G::kIR * G::kIC;
template <bool Xor>
constexpr int input_pitch = Xor ? kKc : kKc + 8;
template <bool Xor>
constexpr int kFusedSmem = kCells * (input_pitch<Xor> * sizeof(__half) + (Xor ? 24 : 8));

struct CompactTap {
  unsigned anchor;
  float wx, wy;
};
static_assert(sizeof(CompactTap) == 12);

__device__ __forceinline__ CompactTap pack_tap(const Tap& t) {
  const unsigned next_x = t.i01 != t.i00, next_y = t.i10 != t.i00;
  return {static_cast<unsigned>(t.i00) | (next_x << 30U) | (next_y << 31U), t.wx, t.wy};
}

__device__ __forceinline__ Tap unpack_tap(const CompactTap& packed, int width) {
  const int base = static_cast<int>(packed.anchor & ((1U << 30U) - 1U));
  const int dx = static_cast<int>((packed.anchor >> 30U) & 1U);
  const int dy = static_cast<int>(packed.anchor >> 31U) * width;
  return {base, base + dx, base + dy, base + dx + dy, packed.wx, packed.wy};
}

// Chaque séquence de huit pixels atteint huit banques vectorielles distinctes.
// Les deux demi-fragments K8 sont échangés au changement du bit 2 du pixel.
__host__ __device__ constexpr int vector_xor(int pixel, int half) {
  return half ^ ((pixel >> 2) & 1);
}

template <bool Xor>
using Motion = std::conditional_t<Xor, CompactTap, uint2>;

template <bool Xor>
__device__ __forceinline__ void store_vectors(__half* tile, int pixel, uint4 a, uint4 b) {
  auto* out = reinterpret_cast<uint4*>(tile + pixel * input_pitch<Xor>);
  if constexpr (Xor) {
    out[vector_xor(pixel, 0)] = a;
    out[vector_xor(pixel, 1)] = b;
  } else {
    out[0] = a;
    out[1] = b;
  }
}

__device__ __forceinline__ bool present(const DrbaGlueParams& g, int x, int y) {
  return x >= 0 && y >= 0 && x < g.width && y < g.height;
}

__device__ __forceinline__ float decode_flow(uint2 four, int c) {
  return __half2float(
      __ushort_as_half(static_cast<unsigned short>((c < 2 ? four.x : four.y) >> (16 * (c & 1)))));
}

__device__ __forceinline__ unsigned half_pair(__half a, __half b) {
  return static_cast<unsigned>(__half_as_ushort(a)) | (static_cast<unsigned>(__half_as_ushort(b)) << 16U);
}

template <bool Xor>
__device__ __forceinline__ void set_flows(Motion<Xor>* flows, const DrbaGlueParams& g, int tx, int ty,
                                          int tid) {
  const SourceZ z{static_cast<const __half*>(g.y), g.height / 2, g.width / 2};
  for (int i = tid; i < kCells; i += rn::kThreads) {
    const int x = tx + i % G::kIC, y = ty + i / G::kIC;
    uint2 packed{};
    if (present(g, x, y)) {
      float r[4];
      enlarge<4>(z, 0, 2, x, y, r);
      packed = {half_pair(__float2half_rn(r[0] * 2.F), __float2half_rn(r[1] * 2.F)),
                half_pair(__float2half_rn(r[2] * 2.F), __float2half_rn(r[3] * 2.F))};
    }
    if constexpr (Xor) {
      if (!present(g, x, y)) continue;
      flows[i] = pack_tap(tap_of(x, y, decode_flow(packed, 0), decode_flow(packed, 1), g.width, g.height));
      flows[kCells + i] =
          pack_tap(tap_of(x, y, decode_flow(packed, 2), decode_flow(packed, 3), g.width, g.height));
    } else {
      flows[i] = packed;
    }
  }
}

struct Pixel {
  const __half* x;
  Tap middle, side;
  int area, at, side_image, side_encoding;
  __half time;

  template <int C>
  __device__ __forceinline__ __half channel() const {
    if constexpr (C <= 56) return x[C * area + at];
    if constexpr (C <= 59) return __float2half_rn(sample(x + (3 + C - 57) * area, middle));
    if constexpr (C <= 62) return __float2half_rn(sample(x + (side_image + C - 60) * area, side));
    if constexpr (C <= 78) return __float2half_rn(sample(x + (25 + C - 63) * area, middle));
    if constexpr (C <= 94) return __float2half_rn(sample(x + (side_encoding + C - 79) * area, side));
    return time;
  }
};

template <int Q, int I>
__device__ __forceinline__ uint4 pack_quad(const Pixel& p) {
  constexpr int base = 16 * Q + I * 8;
  return {half_pair(p.channel<base>(), p.channel<base + 1>()),
          half_pair(p.channel<base + 2>(), p.channel<base + 3>()),
          half_pair(p.channel<base + 4>(), p.channel<base + 5>()),
          half_pair(p.channel<base + 6>(), p.channel<base + 7>())};
}

template <bool Xor, int Q>
__device__ __forceinline__ void set_dynamic(__half* tile, const Motion<Xor>* flows, const DrbaGlueParams& g,
                                            int tx, int ty, int tid, float tau, bool before) {
  static_assert(Q >= 3 && Q <= 5);
  const int area = g.width * g.height;
  const auto* src = static_cast<const __half*>(g.x);
  for (int i = tid; i < kCells; i += rn::kThreads) {
    const int col = i % G::kIC, row = i / G::kIC;
    const int x = tx + col, y = ty + row;
    uint4 a{}, b{};
    if (present(g, x, y)) {
      const auto tap = [&](int image) {
        if constexpr (Xor) {
          return unpack_tap(flows[image * kCells + i], g.width);
        } else {
          const auto f = flows[i];
          return tap_of(x, y, decode_flow(f, 2 * image), decode_flow(f, 2 * image + 1), g.width, g.height);
        }
      };
      const int at = y * g.width + x;
      const __half time = g.scalar_time ? __float2half_rn(tau) : src[(g.compact_time ? 57 : 95) * area + at];
      const Pixel p{src, tap(0), tap(1), area, at, before ? 0 : 6, before ? 9 : 41, time};
      a = pack_quad<Q, 0>(p);
      b = pack_quad<Q, 1>(p);
    }
    store_vectors<Xor>(tile, row * G::kIC + rn::smem_col<kS, G::kIC>(col), a, b);
  }
}

template <bool Xor>
__device__ __forceinline__ void set_controls(__half* tile, const Motion<Xor>* flows, const DrbaGlueParams& g,
                                             int tx, int ty, int tid) {
  const SourceZ z{static_cast<const __half*>(g.y), g.height / 2, g.width / 2};
  for (int i = tid; i < kCells; i += rn::kThreads) {
    const int col = i % G::kIC, row = i / G::kIC;
    const int x = tx + col, y = ty + row;
    uint4 a{}, b{};
    if (present(g, x, y)) {
      if constexpr (Xor) {
        float r[13];
        enlarge<13>(z, 0, 2, x, y, r);
        a = {half_pair(__float2half_rn(r[4]), __float2half_rn(r[5])),
             half_pair(__float2half_rn(r[6]), __float2half_rn(r[7])),
             half_pair(__float2half_rn(r[8]), __float2half_rn(r[9])),
             half_pair(__float2half_rn(r[10]), __float2half_rn(r[11]))};
        b = {half_pair(__float2half_rn(r[12]), __float2half_rn(r[0] * 2.F)),
             half_pair(__float2half_rn(r[1] * 2.F), __float2half_rn(r[2] * 2.F)),
             half_pair(__float2half_rn(r[3] * 2.F), __ushort_as_half(0)), 0U};
      } else {
        float r[9];
        enlarge<9>(z, 4, 2, x, y, r);
        a = {half_pair(__float2half_rn(r[0]), __float2half_rn(r[1])),
             half_pair(__float2half_rn(r[2]), __float2half_rn(r[3])),
             half_pair(__float2half_rn(r[4]), __float2half_rn(r[5])),
             half_pair(__float2half_rn(r[6]), __float2half_rn(r[7]))};
        const auto f = flows[i];
        b = {static_cast<unsigned>(__half_as_ushort(__float2half_rn(r[8]))) | (f.x << 16U),
             (f.x >> 16U) | (f.y << 16U), f.y >> 16U, 0U};
      }
    }
    store_vectors<Xor>(tile, row * G::kIC + rn::smem_col<kS, G::kIC>(col), a, b);
  }
}

template <bool Prefetch = true>
__device__ __forceinline__ void accumulate_xor(rn::Frag<rn::Acc::kF32> (&acc)[2][6], const __half* tile,
                                               const uint4* w, int q, int warp, int lane) {
  constexpr int groups = kCt / 16, chunks = 7;
  const std::uint32_t base = rn::smem_u32(tile);
  const auto* wl = w + lane;
  const auto fetch = [&](uint4(&b)[groups], int tap) {
    const long ks = static_cast<long>(tap) * chunks + q;
#pragma unroll
    for (int n = 0; n < groups; ++n) b[n] = __ldg(wl + (ks * groups + n) * 32);
  };
  uint4 b[groups];
  if constexpr (Prefetch) fetch(b, 0);
#pragma unroll 1
  for (int tap = 0; tap < 9; ++tap) {
    uint4 next[groups];
    if constexpr (Prefetch) {
      if (tap < 8) fetch(next, tap + 1);
    } else {
      fetch(b, tap);
    }
    const int dy = tap / 3, dx = tap % 3;
    const int col0 = dx == 1 ? G::kEven : dx / 2;
    std::uint32_t a[2][4];
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      const int pixel = (2 * warp + dy) * G::kIC + 16 * f + col0 + (lane & 15);
      const int half = vector_xor(pixel, lane >> 4);
      rn::ldmatrix4(a[f], base + static_cast<std::uint32_t>((pixel * input_pitch<true> + half * 8) * 2));
    }
#pragma unroll
    for (int f = 0; f < 2; ++f) {
#pragma unroll
      for (int n = 0; n < groups; ++n) {
        acc[f][2 * n].mma(a[f], b[n].x, b[n].y);
        acc[f][2 * n + 1].mma(a[f], b[n].z, b[n].w);
      }
    }
    if (Prefetch && tap < 8)
#pragma unroll
      for (int n = 0; n < groups; ++n) b[n] = next[n];
  }
}

// Les plans couvrent une largeur paire et x0 est pair. Charger les pixels
// par paires laisse seulement le bord gauche impair à lire séparément.
template <bool Xor>
__device__ __forceinline__ void load_x2_planes(__half* tile, const __half* source, int width, int height,
                                               int channels, int q, int x0, int y0) {
  constexpr int pairs = (G::kIC - 1) / 2;
  const int tid = static_cast<int>(threadIdx.x), area = width * height;
  const auto dst = [tile](int r, int col, int c) -> __half& {
    const int pixel = r * G::kIC + col;
    const int lane = Xor ? vector_xor(pixel, c / 8) * 8 + c % 8 : c;
    return tile[pixel * input_pitch<Xor> + lane];
  };
  for (int i = tid; i < channels * G::kIR * pairs; i += rn::kThreads) {
    const int c = i / (G::kIR * pairs), r = (i / pairs) % G::kIR, col = i % pairs;
    const int x = x0 + 2 * col, y = y0 + r;
    const bool valid = y >= 0 && y < height && x < width;
    const auto* at = valid ? source + (q * kKc + c) * area + y * width + x : source;
    const __half2 pair = valid ? __ldg(reinterpret_cast<const __half2*>(at)) : __floats2half2_rn(0.F, 0.F);
    dst(r, G::kEven + col, c) = __low2half(pair);
    dst(r, col + 1, c) = __high2half(pair);
  }
  for (int i = tid; i < channels * G::kIR; i += rn::kThreads) {
    const int c = i % channels, r = i / channels, x = x0 - 1, y = y0 + r;
    const bool valid = y >= 0 && y < height && x >= 0 && x < width;
    dst(r, 0, c) = valid ? __ldg(source + (q * kKc + c) * area + y * width + x) : __ushort_as_half(0);
  }
}

template <bool Xor, bool Cold = false>
__global__ void __launch_bounds__(rn::kThreads, Xor ? 3 : 2)
    conv_glue(rn::Params p, DrbaGlueParams g, float4* cache) {
  static_assert(!Cold || Xor);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 890
  // Cette fusion est sélectionnée uniquement avec le bloc fin natif Ada.
  __trap();
#else
  extern __shared__ __align__(16) unsigned char raw[];
  auto* const tile = reinterpret_cast<__half*>(raw);
  auto* const flows = reinterpret_cast<Motion<Xor>*>(tile + kCells * input_pitch<Xor>);
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid & 31;
  const int ty = static_cast<int>(blockIdx.y) * G::kTH;
  const int tx = static_cast<int>(blockIdx.x) * rn::kTW;
  const int left = tx * kS - 1, top = ty * kS - 1;
  const auto block = static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  auto* partial = cache + (12 * block * rn::kThreads) + tid;
  constexpr int kChunks = 7;
  rn::Frag<rn::Acc::kF32> acc[2][6];
#pragma unroll
  for (int f = 0; f < 2; ++f)
#pragma unroll
    for (int n = 0; n < 6; ++n) {
      if constexpr (Cold) {
        acc[f][n].clear();
      } else {
        const auto a = __ldg(partial + (6 * f + n) * rn::kThreads);
        acc[f][n].r[0] = a.x;
        acc[f][n].r[1] = a.y;
        acc[f][n].r[2] = a.z;
        acc[f][n].r[3] = a.w;
      }
    }
  if constexpr (Cold) {
#pragma unroll 1
    for (int q = 0; q < 3; ++q) {
      load_x2_planes<true>(tile, static_cast<const __half*>(g.x), g.width, g.height, kKc, q, left + 1, top);
      __syncthreads();
      // La première partie dispose de moins de registres car elle écrit le cache.
      accumulate_xor<false>(acc, tile, p.w, q, warp, lane);
      __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int n = 0; n < 6; ++n) {
        const auto& a = acc[f][n].r;
        partial[(6 * f + n) * rn::kThreads] = make_float4(a[0], a[1], a[2], a[3]);
      }
  }
  set_flows<Xor>(flows, g, left, top, tid);
  __syncthreads();

  const auto full_area = static_cast<std::size_t>(g.full_width) * g.full_height;
  const float tau = 2.F * __half2float(static_cast<const __half*>(g.pair01)[6U * full_area]);
  const bool before = before_side(tau);
#pragma unroll 1
  for (int q = 3; q < kChunks; ++q) {
    if (q != 3) __syncthreads();
    switch (q) {
      case 3:
        set_dynamic<Xor, 3>(tile, flows, g, left, top, tid, tau, before);
        break;
      case 4:
        set_dynamic<Xor, 4>(tile, flows, g, left, top, tid, tau, before);
        break;
      case 5:
        set_dynamic<Xor, 5>(tile, flows, g, left, top, tid, tau, before);
        break;
      default:
        set_controls<Xor>(tile, flows, g, left, top, tid);
    }
    __syncthreads();
    if constexpr (Xor)
      accumulate_xor(acc, tile, p.w, q, warp, lane);
    else
      rn::accumulate<kKc, kCt, kKh, kS, kRows, rn::Acc::kF32, false>(acc, tile, p.w, q, kChunks, 0, 9, warp,
                                                                     lane);
  }
  __syncthreads();
  rn::stage<kCt, kRows, rn::Acc::kF32>(acc, tile, p.bias, warp, lane);
  __syncthreads();
  rn::store<kCt, kRows, rn::Out::kNhwc>(tile, p, ty, tx, 0, 0);
#endif
}
}  // namespace
}  // namespace kirari::gpu::cuda::drba_hot
