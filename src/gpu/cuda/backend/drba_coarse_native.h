#pragma once

#include "gpu/cuda/backend/rife_native_net.h"

namespace kirari::gpu::cuda {

// Les blocs B0 et B1 de DistilDRBA à 1920 × 1088 / 4. Les ports X/Y
// restent planaires comme la colle qui précède et suit les blocs.
class DrbaCoarseNative {
 public:
  virtual ~DrbaCoarseNative() = default;

  static std::unique_ptr<DrbaCoarseNative> create(int block, std::span<const RifeNativeConv> recipe,
                                                  cudaStream_t stream, cudaError_t& error);
  virtual cudaError_t enqueue(cudaStream_t stream, const void* input, void* output,
                              bool same_images) const = 0;
};

}  // namespace kirari::gpu::cuda
