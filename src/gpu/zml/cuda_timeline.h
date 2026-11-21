#pragma once
#include "gpu/vulkan/compute/refuse.h"
#include "media/cuda_driver.h"

namespace kirari::gpu::zml {
// Sémaphore timeline du rendu, importé dans le contexte primaire du GPU.
class CudaTimeline {
 public:
  static std::expected<std::shared_ptr<CudaTimeline>, Refusal> import(
      std::shared_ptr<media::CudaDriver> driver, media::PrimaryContext ctx, int fd);
  virtual ~CudaTimeline() = default;
  virtual vk::Done wait(std::uintptr_t stream, std::uint64_t value) = 0;
  virtual vk::Done signal(std::uintptr_t stream, std::uint64_t value) = 0;
};
}  // namespace kirari::gpu::zml
