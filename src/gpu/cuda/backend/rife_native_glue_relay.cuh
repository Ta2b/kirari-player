#pragma once

#include "gpu/cuda/backend/rife_glue.cuh"
#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::rife_native::relay {
// Disposition CRD pour lire les quatre composantes du flot, ou les huit attributs, avec
// un seul chargement partagé par voisin. La transposition ne change aucun code FP16.
struct RelayCrd {
  const uint2* flow;
  const uint4* features;
  const __half* mask;
  int row0, col0, cols;

  __device__ __forceinline__ int index(int y, int x) const {
    return 4 * (((y >> 1) - row0) * cols + ((x >> 1) - col0)) + 2 * (y & 1) + (x & 1);
  }
};

template <int Parity>
__device__ __forceinline__ std::uint32_t relay_pair(uint2 a, uint2 b) {
  return __byte_perm(Parity < 2 ? a.x : a.y, Parity < 2 ? b.x : b.y, (Parity & 1) == 0 ? 0x5410 : 0x7632);
}

template <int Parity>
__device__ __forceinline__ void relay_scatter(uint2* flow, uint4* features, __half* mask, int cell,
                                              const uint2* from, bool last) {
  const int at = 4 * cell + Parity;
  flow[at] = make_uint2(relay_pair<Parity>(from[0], from[1]), relay_pair<Parity>(from[2], from[3]));
  if (last) {
    mask[at] = __ushort_as_half(
        static_cast<unsigned short>((Parity < 2 ? from[4].x : from[4].y) >> ((Parity & 1) * 16)));
    features[at] = make_uint4(relay_pair<Parity>(from[5], from[6]), relay_pair<Parity>(from[7], from[8]),
                              relay_pair<Parity>(from[9], from[10]), relay_pair<Parity>(from[11], from[12]));
  }
}

__device__ __forceinline__ RelayCrd stage_relay_crd(const FusedFlowSource& f, bool last, warp::Span px,
                                                    warp::Span py, __half* to, int scale) {
  const float inv = 1.F / static_cast<float>(scale);
  const warp::Span rx = warp::coarse_span(px.lo, px.hi, inv, 2 * f.width);
  const warp::Span ry = warp::coarse_span(py.lo, py.hi, inv, 2 * f.height);
  const int row0 = ry.lo / 2, col0 = rx.lo / 2, rows = ry.hi / 2 - row0 + 1, cols = rx.hi / 2 - col0 + 1;
  const int cells = rows * cols;
  auto* flow = reinterpret_cast<uint2*>(to);
  auto* features = last ? reinterpret_cast<uint4*>(to + 16 * cells) : nullptr;
  auto* mask = last ? to + 48 * cells : nullptr;
  const auto* y = static_cast<const __half*>(f.y);
  for (int cell = static_cast<int>(threadIdx.x); cell < cells; cell += kThreads) {
    const int row = row0 + cell / cols, col = col0 + cell % cols;
    const auto* from = reinterpret_cast<const uint2*>(y + (static_cast<long>(row) * f.width + col) * f.pitch);
    uint2 load[13];
#pragma unroll
    for (int c = 0; c < 4; ++c) load[c] = __ldg(from + c);
    if (last) {
#pragma unroll
      for (int c = 4; c < 13; ++c) load[c] = __ldg(from + c);
    }
    relay_scatter<0>(flow, features, mask, cell, load, last);
    relay_scatter<1>(flow, features, mask, cell, load, last);
    relay_scatter<2>(flow, features, mask, cell, load, last);
    relay_scatter<3>(flow, features, mask, cell, load, last);
  }
  return {flow, features, mask, row0, col0, cols};
}

struct FlowQuad {
  int a, b, d, e;
};

__device__ __forceinline__ FlowQuad relay_quad(const RelayCrd& r, const warp::Axis& x, const warp::Axis& y) {
  return {r.index(y.i0, x.i0), r.index(y.i0, x.i1), r.index(y.i1, x.i0), r.index(y.i1, x.i1)};
}

__device__ __forceinline__ float2 relay_resize(std::uint32_t a, std::uint32_t b, std::uint32_t d,
                                               std::uint32_t e, const warp::Axis& x, const warp::Axis& y) {
  const float2 aa = half_pair(a), bb = half_pair(b), dd = half_pair(d), ee = half_pair(e);
  return {warp::rounded(warp::lerp2(aa.x, bb.x, dd.x, ee.x, x, y)),
          warp::rounded(warp::lerp2(aa.y, bb.y, dd.y, ee.y, x, y))};
}

template <int K, bool Fixed>
__device__ __forceinline__ void flow_sample(const FusedGlueParams& g, const warp::Staged (&src)[K], int x,
                                            int y, float (&f)[4], warp::Axis& ax, warp::Axis& ay) {
  glue::flow_at<K, Fixed>(g, src, x, y, f, ax, ay);
}

template <int K, bool Fixed>
__device__ __forceinline__ void flow_sample(const FusedGlueParams& g, const RelayCrd (&src)[K], int x, int y,
                                            float (&f)[4], warp::Axis& ax, warp::Axis& ay) {
#pragma unroll
  for (int j = 0; j < K; ++j) {
    const auto& source = g.source[j];
    const float scale = static_cast<float>(Fixed ? glue::last_scales<K>(j) : source.scale);
    ax = warp::coarse_axis(x, 1.F / scale, 2 * source.width);
    ay = warp::coarse_axis(y, 1.F / scale, 2 * source.height);
    const auto q = relay_quad(src[j], ax, ay);
    const auto* flow = src[j].flow;
    const uint2 a = flow[q.a], b = flow[q.b], d = flow[q.d], e = flow[q.e];
    const auto low = relay_resize(a.x, b.x, d.x, e.x, ax, ay);
    const auto high = relay_resize(a.y, b.y, d.y, e.y, ax, ay);
    const float v[4]{low.x, low.y, high.x, high.y};
#pragma unroll
    for (int c = 0; c < 4; ++c) f[c] = j == 0 ? v[c] * scale : warp::rounded(f[c] + v[c] * scale);
  }
}

__device__ __forceinline__ float relay_mask(const warp::Staged& src, const warp::Axis& x,
                                            const warp::Axis& y) {
  return glue::z_enlarge(src, 4, x, y);
}

__device__ __forceinline__ float relay_mask(const RelayCrd& src, const warp::Axis& x, const warp::Axis& y) {
  const auto q = relay_quad(src, x, y);
  const auto* mask = src.mask;
  return warp::rounded(warp::lerp2(__half2float(mask[q.a]), __half2float(mask[q.b]), __half2float(mask[q.d]),
                                   __half2float(mask[q.e]), x, y));
}

__device__ __forceinline__ void relay_attributes(const warp::Staged& src, const warp::Axis& x,
                                                 const warp::Axis& y, float (&v)[16]) {
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = glue::z_enlarge(src, 5 + k, x, y);
}

__device__ __forceinline__ void relay_attributes(const RelayCrd& src, const warp::Axis& x,
                                                 const warp::Axis& y, float (&v)[16]) {
  const auto q = relay_quad(src, x, y);
  const auto* features = src.features;
  const uint4 a = features[q.a], b = features[q.b], d = features[q.d], e = features[q.e];
  const float2 p0 = relay_resize(a.x, b.x, d.x, e.x, x, y), p1 = relay_resize(a.y, b.y, d.y, e.y, x, y);
  const float2 p2 = relay_resize(a.z, b.z, d.z, e.z, x, y), p3 = relay_resize(a.w, b.w, d.w, e.w, x, y);
  v[0] = p0.x;
  v[1] = p0.y;
  v[2] = p1.x;
  v[3] = p1.y;
  v[4] = p2.x;
  v[5] = p2.y;
  v[6] = p3.x;
  v[7] = p3.y;
}

}  // namespace kirari::gpu::cuda::rife_native::relay
