// La colle X2 fournit les quatre tranches mobiles à la convolution du dernier bloc.
#pragma once
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
constexpr int kKc = 16, kCt = 48, kRows = 1, kS = 2, kKh = 3, kLds = kKc + 8;
using G = rn::Geometry<kKh, kS, kRows>;
constexpr int kCells = G::kIR * G::kIC;
constexpr int kFusedSmem = kCells * (kLds * sizeof(__half) + sizeof(uint2));

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

__device__ __forceinline__ void set_flows(uint2* flows, const DrbaGlueParams& g, int tx, int ty, int tid) {
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
    flows[i] = packed;
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

template <int Q>
__device__ __forceinline__ void set_dynamic(__half* tile, const uint2* flows, const DrbaGlueParams& g, int tx,
                                            int ty, int tid, float tau, bool before) {
  static_assert(Q >= 3 && Q <= 5);
  const int area = g.width * g.height;
  const auto* src = static_cast<const __half*>(g.x);
  for (int i = tid; i < kCells; i += rn::kThreads) {
    const int col = i % G::kIC, row = i / G::kIC;
    const int x = tx + col, y = ty + row;
    uint4 a{}, b{};
    if (present(g, x, y)) {
      const auto f = flows[i];
      const int at = y * g.width + x;
      const __half time = g.scalar_time ? __float2half_rn(tau) : src[(g.compact_time ? 57 : 95) * area + at];
      const Pixel p{src,
                    tap_of(x, y, decode_flow(f, 0), decode_flow(f, 1), g.width, g.height),
                    tap_of(x, y, decode_flow(f, 2), decode_flow(f, 3), g.width, g.height),
                    area,
                    at,
                    before ? 0 : 6,
                    before ? 9 : 41,
                    time};
      a = pack_quad<Q, 0>(p);
      b = pack_quad<Q, 1>(p);
    }
    auto* dest = reinterpret_cast<uint4*>(tile + (row * G::kIC + rn::smem_col<kS, G::kIC>(col)) * kLds);
    dest[0] = a;
    dest[1] = b;
  }
}

__device__ __forceinline__ void set_controls(__half* tile, const uint2* flows, const DrbaGlueParams& g,
                                             int tx, int ty, int tid) {
  const SourceZ z{static_cast<const __half*>(g.y), g.height / 2, g.width / 2};
  for (int i = tid; i < kCells; i += rn::kThreads) {
    const int col = i % G::kIC, row = i / G::kIC;
    const int x = tx + col, y = ty + row;
    uint4 a{}, b{};
    if (present(g, x, y)) {
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
    auto* dest = reinterpret_cast<uint4*>(tile + (row * G::kIC + rn::smem_col<kS, G::kIC>(col)) * kLds);
    dest[0] = a;
    dest[1] = b;
  }
}

__global__ void __launch_bounds__(rn::kThreads, 2)
    conv_glue_hot(rn::Params p, DrbaGlueParams g, const float4* cache) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 890
  // Cette fusion est sélectionnée uniquement avec le bloc fin natif Ada.
  __trap();
#else
  extern __shared__ __align__(16) unsigned char raw[];
  auto* const tile = reinterpret_cast<__half*>(raw);
  auto* const flows = reinterpret_cast<uint2*>(tile + kCells * kLds);
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid & 31;
  const int ty = static_cast<int>(blockIdx.y) * G::kTH;
  const int tx = static_cast<int>(blockIdx.x) * rn::kTW;
  const int left = tx * kS - 1, top = ty * kS - 1;
  set_flows(flows, g, left, top, tid);
  __syncthreads();

  const auto full_area = static_cast<std::size_t>(g.full_width) * g.full_height;
  const float tau = 2.F * __half2float(static_cast<const __half*>(g.pair01)[6U * full_area]);
  const bool before = before_side(tau);
  const auto block = static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  const auto* partial = cache + (12 * block * rn::kThreads) + tid;
  rn::Frag<rn::Acc::kF32> acc[2][6];
#pragma unroll
  for (int f = 0; f < 2; ++f)
#pragma unroll
    for (int n = 0; n < 6; ++n) {
      const auto a = __ldg(partial + (6 * f + n) * rn::kThreads);
      acc[f][n].r[0] = a.x;
      acc[f][n].r[1] = a.y;
      acc[f][n].r[2] = a.z;
      acc[f][n].r[3] = a.w;
    }
  constexpr int kChunks = 7;
#pragma unroll 1
  for (int q = 3; q < kChunks; ++q) {
    if (q != 3) __syncthreads();
    switch (q) {
      case 3:
        set_dynamic<3>(tile, flows, g, left, top, tid, tau, before);
        break;
      case 4:
        set_dynamic<4>(tile, flows, g, left, top, tid, tau, before);
        break;
      case 5:
        set_dynamic<5>(tile, flows, g, left, top, tid, tau, before);
        break;
      default:
        set_controls(tile, flows, g, left, top, tid);
    }
    __syncthreads();
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
