#include <array>
#include <cstddef>

#include "gpu/cuda/backend/drba_coarse_kernels.cuh"
#include "gpu/cuda/backend/drba_coarse_native.h"
#include "gpu/cuda/backend/rife_native_layer.h"

namespace kirari::gpu::cuda {
namespace {

namespace rn = rife_native;
namespace coarse = drba_coarse_kernels;
using coarse::kAcc;
using coarse::kCt;
using coarse::kKc;
using coarse::kPhaseRows;
using coarse::kRows;
constexpr int kOutput = 52, kOutputPitch = 64;
constexpr int kLayers = 11;

struct Layer {
  rn::Buffer weights, biases, slopes, scales;

  cudaError_t setup(RifeNativeConv recipe, rn::Kind kind, int cin, cudaStream_t stream) {
    const auto beta = std::move(recipe.scale);
    recipe.scale.clear();
    const auto dense = rn::dense_weights(recipe, kind, cin);
    const auto packed = rn::packed_weights(dense, kCt);
    auto result = weights.upload(packed.data(), packed.size() * sizeof(std::uint32_t), stream);
    if (result == cudaSuccess)
      result = biases.upload(dense.bias.data(), dense.bias.size() * sizeof(float), stream);
    if (result == cudaSuccess && !recipe.slope.empty())
      result = slopes.upload(recipe.slope.data(), recipe.slope.size() * sizeof(float), stream);
    if (result == cudaSuccess && !beta.empty())
      result = scales.upload(beta.data(), beta.size() * sizeof(float), stream);
    const auto synced = cudaStreamSynchronize(stream);
    return result == cudaSuccess ? synced : result;
  }

  rn::Params bind(const __half* input, __half* output, int ih, int iw, int h, int w, int cin, int cout,
                  int pitch) const {
    return {.in = input,
            .in_ld = cin,
            .ih = ih,
            .iw = iw,
            .out = output,
            .out_ld = pitch,
            .out_ch = cout,
            .oh = h,
            .ow = w,
            .w = weights.as<const uint4>(),
            .bias = biases.as<const float>(),
            .slope = slopes.as<const float>(),
            .cin = cin,
            .tiles = pitch / kCt};
  }
};

bool matches_layer(const RifeNativeConv& layer, int cin, int cout, bool transposed, bool residual,
                   int stride) {
  const auto count = static_cast<std::size_t>(cout);
  const auto taps = static_cast<std::size_t>(transposed ? 16 : 9);
  return layer.cin == cin && layer.cout == cout && layer.transposed == transposed && layer.stride == stride &&
         layer.weights.size() == taps * static_cast<std::size_t>(cin) * count && layer.bias.size() == count &&
         layer.scale.size() == (residual ? count : 0) && layer.slope.size() == (transposed ? 0 : count);
}

template <int Block>
class Native final : public DrbaCoarseNative {
 public:
  cudaError_t setup(std::span<const RifeNativeConv> recipe, cudaStream_t stream) {
    if (recipe.size() != layers_.size()) return cudaErrorInvalidValue;
    int device = 0;
    cudaDeviceProp properties{};
    if (auto error = cudaGetDevice(&device); error != cudaSuccess) return error;
    if (auto error = cudaGetDeviceProperties(&properties, device); error != cudaSuccess) return error;
    ada_60_sm_ = properties.major == 8 && properties.minor == 9 && properties.multiProcessorCount == 60;
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      const bool first = i == 0, stride = i <= 1, last = i + 1 == layers_.size();
      const bool residual = !stride && !last;
      const int cin = first ? kInput : stride ? kMid : kChannels;
      const int cout = first ? kMid : last ? kOutput : kChannels;
      const auto& layer = recipe[i];
      if (!matches_layer(layer, cin, cout, last, residual, (stride || last) ? 2 : 1))
        return cudaErrorInvalidValue;
      const auto kind = last ? rn::Kind::kPhase : stride ? rn::Kind::kHalf : rn::Kind::kSame;
      if (const auto result = layers_[i].setup(layer, kind, first ? kInputPitch : cin, stream);
          result != cudaSuccess)
        return result;
    }
    if constexpr (Block == 0) {
      constexpr auto grid = tiles(kH / 2, kW / 2, kRows, kMid / kCt);
      const auto status =
          partial_.allocate(sizeof(float4) * grid.x * grid.y * grid.z * kRows * 2 * (kCt / 8) * rn::kThreads);
      if (status != cudaSuccess) return status;
    }
    return workspace_.allocate(sizeof(__half) * (kMidSize + 2 * kBodySize));
  }

  cudaError_t enqueue(cudaStream_t stream, const void* input, void* output, bool same_images) const override {
    auto* mid = workspace_.as<__half>();
    auto* state = mid + kMidSize;
    auto* next = state + kBodySize;
    constexpr int stride_smem = rn::smem_bytes(kKc, kCt, 3, 2, kRows);
    const auto first = layers_[0].bind(static_cast<const __half*>(input), mid, kH, kW, kH / 2, kW / 2,
                                       kInputPitch, kMid, kMid);
    constexpr auto grid = tiles(kH / 2, kW / 2, kRows, kMid / kCt);
    if constexpr (Block == 0) {
      if (same_images)
        coarse::entry_grouped<kW, kH, kInput, kInputPitch, true>
            <<<grid, rn::kThreads, stride_smem, stream>>>(first, partial_.as<float4>());
      else
        coarse::entry_grouped<kW, kH, kInput, kInputPitch>
            <<<grid, rn::kThreads, stride_smem, stream>>>(first, partial_.as<float4>());
    } else {
      coarse::entry_grouped<kW, kH, kInput, kInputPitch><<<grid, rn::kThreads, stride_smem, stream>>>(first);
    }
    if (const auto result = cudaGetLastError(); result != cudaSuccess) return result;
    const auto second =
        layers_[1].bind(mid, state, kH / 2, kW / 2, kH / 4, kW / 4, kMid, kChannels, kChannels);
    rn::conv_kernel<kKc, kCt, 3, 2, kRows, rn::In::kNhwc, rn::Out::kNhwc, kAcc>
        <<<tiles(kH / 4, kW / 4, kRows, kChannels / kCt), rn::kThreads, stride_smem, stream>>>(second);
    if (const auto result = cudaGetLastError(); result != cudaSuccess) return result;
    for (std::size_t i = 2; i + 1 < layers_.size(); ++i) {
      const auto& layer = layers_[i];
      const auto p = layer.bind(state, next, kH / 4, kW / 4, kH / 4, kW / 4, kChannels, kChannels, kChannels);
      const auto* scale = layer.scales.template as<const float>();
      const auto* slope = layer.slopes.template as<const float>();
      if (ada_60_sm_)
        coarse::launch<kChannels, kCt, 4, 1, 9>(p, scale, slope, stream);
      else
        coarse::launch<kChannels, kCt, 2, 1>(p, scale, slope, stream);
      if (const auto result = cudaGetLastError(); result != cudaSuccess) return result;
      std::swap(state, next);
    }
    const auto final = layers_.back().bind(state, static_cast<__half*>(output), kH / 4, kW / 4, kH / 4,
                                           kW / 4, kChannels, kOutput, kOutputPitch);
    constexpr int phase_smem = rn::smem_bytes(kKc, kCt, 2, 1, kPhaseRows);
    coarse::project_planar<<<tiles(kH / 4, kW / 4, kPhaseRows, 4 * kOutputPitch / kCt), rn::kThreads,
                             phase_smem, stream>>>(final);
    return cudaGetLastError();
  }

 private:
  static_assert(Block == 0 || Block == 1);
  static constexpr int kW = Block == 0 ? 120 : 240, kH = Block == 0 ? 68 : 136;
  static constexpr int kInput = Block == 0 ? 58 : 109;
  static constexpr int kInputPitch = (kInput + 15) / 16 * 16;
  static constexpr int kMid = Block == 0 ? 96 : 64, kChannels = 2 * kMid;
  static constexpr int kMidSize = kH / 2 * (kW / 2) * kMid, kBodySize = kH / 4 * (kW / 4) * kChannels;

  static constexpr dim3 tiles(int h, int w, int rows, int z) {
    const int height = rows * rn::kWarps;
    return dim3((w + rn::kTW - 1) / rn::kTW, (h + height - 1) / height, z);
  }

  std::array<Layer, kLayers> layers_;
  rn::Buffer workspace_, partial_;
  bool ada_60_sm_ = false;
};

template <int Block>
std::unique_ptr<DrbaCoarseNative> create_native(std::span<const RifeNativeConv> recipe, cudaStream_t stream,
                                                cudaError_t& error) {
  auto native = std::make_unique<Native<Block>>();
  error = native->setup(recipe, stream);
  return error == cudaSuccess ? std::move(native) : nullptr;
}

}  // namespace

std::unique_ptr<DrbaCoarseNative> DrbaCoarseNative::create(int block, std::span<const RifeNativeConv> recipe,
                                                           cudaStream_t stream, cudaError_t& error) {
  switch (block) {
    case 0:
      return create_native<0>(recipe, stream, error);
    case 1:
      return create_native<1>(recipe, stream, error);
    default:
      error = cudaErrorInvalidValue;
      return {};
  }
}

}  // namespace kirari::gpu::cuda
