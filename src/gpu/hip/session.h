#pragma once

#include <array>
#include <filesystem>
#include <mutex>

#include "gpu/vulkan/backend/net_runtime.h"
#include "gpu/vulkan/compute/refuse.h"

namespace kirari::gpu::hip_backend {

class Session {
 public:
  enum Kernel : std::size_t { Unshuffle, Stem, Relu, Plain, Gate, Project, Pixel, Count };
  static constexpr std::array<const char*, Count> names{"unshuffle", "stem",    "relu",         "plain",
                                                        "gate",      "project", "pixel_shuffle"};

  virtual ~Session() = default;
  std::mutex sequence;

  virtual std::string_view architecture() const = 0;
  virtual vk::Done in_context(const std::function<void()>& work) = 0;
  virtual void* allocate(std::size_t bytes, std::string_view name) = 0;
  virtual void release(void* pointer) noexcept = 0;
  virtual void upload(void* destination, const void* source, std::size_t bytes, std::string_view name) = 0;
  virtual void download(void* destination, const void* source, std::size_t bytes) = 0;
  virtual void launch(Kernel which, unsigned x, unsigned y, std::span<void*> parameters) = 0;
  virtual void synchronize() = 0;
  virtual bool shares_memory() const = 0;
  virtual std::expected<vk::net::DeviceMemory, Refusal> import_memory(std::uintptr_t, std::size_t,
                                                                      std::size_t);

 protected:
  Session() = default;
};

#ifdef KIRARI_HIP_ROCM
std::expected<std::shared_ptr<Session>, Refusal> make_rocm(const vk::DeviceCaps&,
                                                           const std::filesystem::path&);
#endif
#ifdef KIRARI_HIP_CUDA
std::expected<std::shared_ptr<Session>, Refusal> make_cuda(const vk::DeviceCaps&,
                                                           const std::filesystem::path&);
#endif
std::unexpected<Refusal> refuse_hip(RefusalCode, std::string_view);

}  // namespace kirari::gpu::hip_backend
