#pragma once

#include <cstdint>
#include <string_view>

namespace kirari::gpu {

// Les deux sélections partagent les kernels de la chaîne et le module NVIDIA.
inline constexpr std::uint32_t kCudaNativeInference = 1U << 29U;
inline constexpr std::uint32_t kTensorRtInference = 1U << 30U;
inline constexpr std::uint32_t kCudaInferenceModes = kCudaNativeInference | kTensorRtInference;

constexpr bool is_cuda_backend(std::string_view name) {
  return name == "cuda" || name == "tensorrt";
}

}  // namespace kirari::gpu
