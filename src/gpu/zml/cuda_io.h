#pragma once
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/zml/pjrt.h"

namespace kirari::gpu::zml {
class CudaIo {
 public:
  virtual ~CudaIo() = default;
  static std::shared_ptr<CudaIo> open(Client& client, const vk::DeviceUuid& uuid);
  virtual std::expected<vk::net::DeviceMemory, Refusal> import(std::uintptr_t handle, std::size_t allocation,
                                                               std::size_t bytes) = 0;
  virtual vk::Done attach(int fd) = 0;
  virtual vk::Done wait(std::uint64_t value) = 0;
  virtual vk::Done signal(std::uint64_t value) = 0;
  virtual vk::Done drain() = 0;
  virtual std::expected<PJRT_Buffer*, Refusal> view(Client& client, const vk::net::HostBinding& input,
                                                    PJRT_Buffer_Type type) = 0;
  virtual std::expected<void, Refusal> copy(Client& client, PJRT_Buffer* output,
                                            const vk::net::HostBinding& target,
                                            std::vector<std::shared_ptr<void>>* keep) = 0;
};
}  // namespace kirari::gpu::zml
