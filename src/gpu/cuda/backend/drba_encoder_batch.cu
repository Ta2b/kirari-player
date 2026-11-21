#include <array>
#include <cstddef>
#include <cstdint>

#include "gpu/cuda/backend/drba_encoder_batch.h"
#include "gpu/cuda/backend/drba_encoder_projection.cuh"
#include "gpu/cuda/backend/rife_native_layer.h"

namespace kirari::gpu::cuda {
namespace {

namespace rn = rife_native;
constexpr int kWidth = 480;
constexpr int kHeight = 272;
constexpr int kPixels = kWidth * kHeight;
constexpr int kBatch = 3;
constexpr int kChannels = 16;
constexpr int kIntermediate = kPixels * kChannels / 4;
class BatchEncoder final : public DrbaBatchEncoder {
 public:
  cudaError_t initialize(std::span<const RifeNativeConv> model) {
    if (model.size() != layers_.size()) return cudaErrorInvalidValue;
    constexpr std::array kinds{rn::Kind::kImage, rn::Kind::kSame, rn::Kind::kSame, rn::Kind::kPhase};
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      const bool first = i == 0;
      const bool last = i + 1 == layers_.size();
      const auto& conv = model[i];
      const int input_channels = first ? 3 : kChannels;
      const std::size_t weights_count = (last ? 16U : 9U) * input_channels * kChannels;
      if (conv.transposed != last || conv.stride != (first || last ? 2 : 1) || conv.cin != input_channels ||
          conv.cout != kChannels || conv.weights.size() != weights_count || conv.bias.size() != kChannels ||
          conv.slope.size() != (last ? 0U : kChannels) || !conv.scale.empty())
        return cudaErrorInvalidValue;
      auto& layer = layers_[i];
      layer.kind = kinds[i];
      layer.ih = first ? kHeight : kHeight / 2;
      layer.iw = first ? kWidth : kWidth / 2;
      layer.oh = kHeight / 2;
      layer.ow = kWidth / 2;
      layer.in_ld = first ? 0 : kChannels;
      layer.out_ld = layer.out_ch = layer.cin = kChannels;
      layer.sums = RifeNativeSums::kF32;
      layer.activation = !last;
      const auto dense = rn::dense_weights(conv, layer.kind, layer.cin);
      layer.phases = dense.phases;
      layer.cout = dense.cout;
      const rn::Config config{.kc = kChannels, .ct = kChannels, .rows = first ? 1 : 2, .ksplit = 1};
      if (!rn::shape(layer, config)) return cudaErrorInvalidConfiguration;
      const auto packed = rn::packed_weights(dense, config.ct);
      // Le chargement garde les tableaux hôte en vie jusqu'à la fin des copies.
      if (const auto err = upload(layer, conv, dense.bias, packed); err != cudaSuccess) return err;
    }
    return cudaSuccess;
  }

  cudaError_t enqueue(cudaStream_t stream, const void* input, void* output, void* half, void* quarter,
                      void* scratch) const override {
    if (input == nullptr || output == nullptr || half == nullptr || quarter == nullptr || scratch == nullptr)
      return cudaErrorInvalidValue;
    const auto* in = static_cast<const __half*>(input);
    auto* intermediate = static_cast<__half*>(scratch);
    __half* const buffers[]{intermediate, intermediate + kBatch * kIntermediate};
    for (std::size_t i = 0; i + 1 < layers_.size(); ++i) {
      const auto& layer = layers_[i];
      auto* out = buffers[i % 2];
      rn::BatchParams params{};
      static_cast<rn::Params&>(params) = rn::params(layer, in, out, nullptr, layer.w.as<const void>());
      params.batch = kBatch;
      params.in_batch = i == 0 ? kPixels * 3 : kIntermediate;
      params.out_batch = kIntermediate;
      const dim3 grid(layer.grid.x, layer.grid.y, layer.grid.z * kBatch);
      if (i == 0)
        launch<3, 2, 1, rn::In::kImage, rn::Out::kNhwc>(stream, grid, layer.smem, params);
      else
        launch<3, 1, 2, rn::In::kNhwc, rn::Out::kNhwc>(stream, grid, layer.smem, params);
      if (const auto err = cudaGetLastError(); err != cudaSuccess) return err;
      in = out;
    }
    const auto& final = layers_.back();
    const auto p = rn::params(final, in, static_cast<__half*>(output), nullptr, final.w.as<const void>());
    drba_encoder_projection::launch(p, static_cast<std::uint16_t*>(output), static_cast<std::uint16_t*>(half),
                                    static_cast<std::uint16_t*>(quarter), stream);
    return cudaGetLastError();
  }

 private:
  std::array<rn::Layer, 4> layers_;

  static cudaError_t upload(rn::Layer& layer, const RifeNativeConv& conv, std::span<const float> bias,
                            std::span<const std::uint32_t> weights) {
    auto err = layer.w.upload(weights.data(), weights.size_bytes(), nullptr);
    if (err == cudaSuccess) err = layer.bias.upload(bias.data(), bias.size_bytes(), nullptr);
    if (err == cudaSuccess && layer.activation)
      err = layer.slope.upload(conv.slope.data(), conv.slope.size() * sizeof(float), nullptr);
    const auto sync = cudaStreamSynchronize(nullptr);
    return err == cudaSuccess ? sync : err;
  }

  template <int Kh, int Step, int Rows, rn::In In, rn::Out Out>
  static void launch(cudaStream_t stream, dim3 grid, int smem, const rn::BatchParams& params) {
    rn::conv_kernel<16, 16, Kh, Step, Rows, In, Out, rn::Acc::kF32, false, true>
        <<<grid, rn::kThreads, static_cast<std::size_t>(smem), stream>>>(params);
  }
};

}  // namespace

std::size_t DrbaBatchEncoder::scratch_bytes() {
  return 2U * kBatch * sizeof(__half) * kIntermediate;
}

std::unique_ptr<DrbaBatchEncoder> DrbaBatchEncoder::create(std::span<const RifeNativeConv> model,
                                                           cudaError_t& error) {
  auto encoder = std::make_unique<BatchEncoder>();
  error = encoder->initialize(model);
  if (error != cudaSuccess) return {};
  return encoder;
}

}  // namespace kirari::gpu::cuda
