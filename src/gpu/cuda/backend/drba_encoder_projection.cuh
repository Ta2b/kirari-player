#pragma once

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::drba_encoder_projection {
namespace rn = kirari::gpu::cuda::rife_native;
namespace {
namespace drba_rangement {

inline constexpr int kWidth = 480;
inline constexpr int kHeight = 272;
inline constexpr int kPixels = kWidth * kHeight;
inline constexpr int kImages = 3;
inline constexpr int kChannels = 16;
inline constexpr int kPlanes = kImages * kChannels;
inline constexpr int kTileWidth = 32;
inline constexpr int kTileHeight = 4;
inline constexpr int kThreads = 128;
static_assert(kWidth % kTileWidth == 0 && kHeight % kTileHeight == 0);

__device__ __forceinline__ ushort average(ushort a, ushort b, ushort c, ushort d) {
  // Même ordre de lecture et d'addition que post_planes : haut-gauche, haut-droite, bas-gauche, bas-droite.
  const float sum = __half2float(__ushort_as_half(a)) + __half2float(__ushort_as_half(b)) +
                    __half2float(__ushort_as_half(c)) + __half2float(__ushort_as_half(d));
  return __half_as_ushort(__float2half_rn(sum * 0.25F));
}

__device__ __forceinline__ uint pack(ushort lo, ushort hi) {
  return static_cast<uint>(lo) | (static_cast<uint>(hi) << 16);
}

__device__ __forceinline__ ushort bottom(uint packed) {
  return static_cast<ushort>(packed);
}

__device__ __forceinline__ ushort top(uint packed) {
  return static_cast<ushort>(packed >> 16);
}

__device__ __forceinline__ uint2 planar_row(const uint* row, int stride, int shift) {
  return make_uint2(pack(bottom(row[0] >> shift), bottom(row[stride] >> shift)),
                    pack(bottom(row[2 * stride] >> shift), bottom(row[3 * stride] >> shift)));
}

__device__ __forceinline__ uint half_row(uint2 upper, uint2 lower) {
  return pack(average(bottom(upper.x), top(upper.x), bottom(lower.x), top(lower.x)),
              average(bottom(upper.y), top(upper.y), bottom(lower.y), top(lower.y)));
}

}  // namespace drba_rangement
namespace dr = drba_rangement;
using Sum = rn::Frag<rn::Acc::kF32>;
constexpr int kWidth = 16, kHeight = rn::kWarps, kChannels = dr::kChannels;
constexpr int kInputWidth = kWidth + 2, kInputHeight = kHeight + 2;
constexpr int kInputPitch = kChannels + 8;
constexpr int kTileWidth = kWidth * 2, kTileHeight = kHeight * 2;
// Le pas de neuf mots répartit les canaux d'une rangée sur les banques.
constexpr int kOutputPitch = kChannels + 2;
constexpr int kInputWords = kInputWidth * kInputHeight * kInputPitch;
constexpr int kSharedBytes = (kInputWords + kTileWidth * kTileHeight * kOutputPitch) * sizeof(__half);
static_assert(kTileWidth == 32 && kTileHeight == 8);
static_assert((dr::kWidth / 2) % kWidth == 0 && (dr::kHeight / 2) % kHeight == 0);

__device__ __forceinline__ void sample(Sum (&sum)[4][2], const uint4* w, unsigned base, int lane) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      std::uint32_t a[4];
      rn::ldmatrix4(a, base + ((r * kInputWidth + c) * kInputPitch * 2));
#pragma unroll
      for (int phase = 0; phase < 4; ++phase) {
        const int dy = r - phase / 2, dx = c - phase % 2;
        if (dy >= 0 && dy < 2 && dx >= 0 && dx < 2) {
          const uint4 b = __ldg(w + (4 * phase + 2 * dy + dx) * 32 + lane);
          sum[phase][0].mma(a, b.x, b.y);
          sum[phase][1].mma(a, b.z, b.w);
        }
      }
    }
  }
}

__device__ __forceinline__ void publish(const Sum (&sum)[4][2], const float* bias, __half* tile, int warp,
                                        int lane) {
  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int phase = 0; phase < 4; ++phase) {
    const int y = 2 * warp + phase / 2;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int channel = n * 8 + 2 * t;
      const auto b = __ldg(reinterpret_cast<const float2*>(bias + phase * kChannels + channel));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int x = 2 * (g + 8 * h) + phase % 2;
        const auto v = sum[phase][n].pair(h);
        *reinterpret_cast<__half2*>(tile + (y * kTileWidth + x) * kOutputPitch + channel) =
            rn::to_half2(v.x + b.x, v.y + b.y);
      }
    }
  }
}

__device__ __forceinline__ void arrange(const uint* tile, ushort* full, ushort* half, ushort* quarter, int ox,
                                        int oy, int batch, int thread) {
  constexpr int groups = kTileWidth / 4, stride = kOutputPitch / 2;
  const int col = (thread % groups) * 4;
  const int channel = thread / groups, shift = (channel % 2) * 16;
  const int plane = batch * kChannels + channel, x = ox + col;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int y = oy + i * 4, tile_row = i * 4;
    const auto* row = tile + (tile_row * kTileWidth + col) * stride + channel / 2;
    const uint2 row0 = dr::planar_row(row, stride, shift);
    const uint2 row1 = dr::planar_row(row + kTileWidth * stride, stride, shift);
    const uint2 row2 = dr::planar_row(row + 2 * kTileWidth * stride, stride, shift);
    const uint2 row3 = dr::planar_row(row + 3 * kTileWidth * stride, stride, shift);
    auto* all = reinterpret_cast<uint2*>(full + plane * dr::kPixels + y * dr::kWidth + x);
    all[0] = row0;
    all[dr::kWidth / 4] = row1;
    all[2 * dr::kWidth / 4] = row2;
    all[3 * dr::kWidth / 4] = row3;
    auto* small =
        reinterpret_cast<uint*>(half + plane * (dr::kPixels / 4) + (y / 2) * (dr::kWidth / 2) + x / 2);
    small[0] = dr::half_row(row0, row1);
    small[dr::kWidth / 4] = dr::half_row(row2, row3);
    quarter[plane * (dr::kPixels / 16) + (y / 4) * (dr::kWidth / 4) + x / 4] =
        dr::average(dr::top(row1.x), dr::bottom(row1.y), dr::top(row2.x), dr::bottom(row2.y));
  }
}

__global__ void __launch_bounds__(rn::kThreads, 3)
    project(rn::Params p, ushort* full, ushort* half, ushort* quarter) {
  extern __shared__ __align__(16) unsigned char raw[];
  auto* tile = reinterpret_cast<__half*>(raw);
  auto* output = tile + kInputWords;
  const int id = static_cast<int>(threadIdx.x), lane = id % 32, warp = id / 32;
  const int x = static_cast<int>(blockIdx.x) * kWidth, y = static_cast<int>(blockIdx.y) * kHeight;
  const int batch = static_cast<int>(blockIdx.z);
  p.in += batch * p.ih * p.iw * kChannels;
  rn::load_nhwc<kChannels, 1, kInputHeight, kInputWidth, false>(tile, p, 0, y - 1, x - 1);
  rn::cp_async_wait_all();
  __syncthreads();
  Sum sum[4][2];
#pragma unroll
  for (auto& phase : sum) {
    phase[0].clear();
    phase[1].clear();
  }
  const unsigned base =
      rn::smem_u32(tile) + ((warp * kInputWidth + (lane & 15)) * kInputPitch + (lane >> 4) * 8) * 2;
  sample(sum, p.w, base, lane);
  publish(sum, p.bias, output, warp, lane);
  __syncthreads();
  arrange(reinterpret_cast<const uint*>(output), full, half, quarter, x * 2, y * 2, batch, id);
}

}  // namespace

inline void launch(rn::Params p, ushort* full, ushort* half, ushort* quarter, cudaStream_t stream) {
  const dim3 grid((dr::kWidth / 2) / kWidth, (dr::kHeight / 2) / kHeight, dr::kImages);
  project<<<grid, rn::kThreads, kSharedBytes, stream>>>(p, full, half, quarter);
}
}  // namespace kirari::gpu::cuda::drba_encoder_projection
