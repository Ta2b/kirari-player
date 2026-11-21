// Import d'un tampon de stockage Vulkan dédié par ROCm sous Linux.
#pragma once

#if defined(__linux__)
#include <fcntl.h>
#include <format>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <memory>
#include <unistd.h>
#include <utility>

#include "core/dynamic_library.h"
#include "gpu/vulkan/backend/net_runtime.h"
#include "gpu/vulkan/compute/refuse.h"

namespace kirari::gpu::hip_interop {

struct ExternalMemoryApi {
  decltype(hipImportExternalMemory)* import{};
  decltype(hipExternalMemoryGetMappedBuffer)* map{};
  decltype(hipDestroyExternalMemory)* destroy{};
  decltype(hipFree)* free{};

  static ExternalMemoryApi load(const DynamicLibrary& library) {
    return {library.symbol<decltype(hipImportExternalMemory)>("hipImportExternalMemory").value_or(nullptr),
            library.symbol<decltype(hipExternalMemoryGetMappedBuffer)>("hipExternalMemoryGetMappedBuffer")
                .value_or(nullptr),
            library.symbol<decltype(hipDestroyExternalMemory)>("hipDestroyExternalMemory").value_or(nullptr),
            library.symbol<decltype(hipFree)>("hipFree").value_or(nullptr)};
  }

  explicit operator bool() const { return import && map && destroy && free; }
};

using OnDevice = std::function<vk::Done(const std::function<void()>&)>;

class ExternalBuffer {
 public:
  ExternalBuffer(ExternalMemoryApi api, OnDevice on_device) : api_(api), on_device_(std::move(on_device)) {}
  ExternalBuffer(const ExternalBuffer&) = delete;
  ExternalBuffer& operator=(const ExternalBuffer&) = delete;

  ~ExternalBuffer() {
    try {
      if (memory_ != nullptr) {
        static_cast<void>(on_device_([this] {
          if (address_ != nullptr) static_cast<void>(api_.free(address_));
          static_cast<void>(api_.destroy(memory_));
        }));
      }
    } catch (...) {
      // Le pilote peut avoir perdu l’appareil pendant la fermeture.
    }
    if (fd_ >= 0) ::close(fd_);
  }

  vk::Done open(std::uintptr_t fd, std::size_t allocation, std::size_t bytes) {
    if (!api_ || fd > std::numeric_limits<int>::max() || bytes == 0 || bytes > allocation)
      return refuse(RefusalCode::FeatureUnavailable, "HIP : taille ou descripteur Vulkan invalide");
    // Le producteur Vulkan ferme l’original. ROCm peut relire le descripteur au mappage.
    fd_ = ::fcntl(static_cast<int>(fd), F_DUPFD_CLOEXEC, 0);
    if (fd_ < 0) return refuse(RefusalCode::FeatureUnavailable, "HIP : duplication du FD Vulkan");
    hipError_t status = hipSuccess;
    const char* operation = "hipImportExternalMemory";
    auto selected = on_device_([&] {
      const hipExternalMemoryHandleDesc description{.type = hipExternalMemoryHandleTypeOpaqueFd,
                                                    .handle = {.fd = fd_},
                                                    .size = allocation,
                                                    .flags = hipExternalMemoryDedicated};
      status = api_.import(&memory_, &description);
      if (status != hipSuccess) return;
      operation = "hipExternalMemoryGetMappedBuffer";
      const hipExternalMemoryBufferDesc region{.offset = 0, .size = bytes, .flags = 0};
      status = api_.map(&address_, memory_, &region);
    });
    if (!selected) return selected;
    if (status != hipSuccess || address_ == nullptr)
      return refuse(RefusalCode::FeatureUnavailable,
                    std::format("HIP : {} ({})", operation, static_cast<int>(status)));
    return {};
  }

  std::uintptr_t address() const { return reinterpret_cast<std::uintptr_t>(address_); }

 private:
  ExternalMemoryApi api_;
  OnDevice on_device_;
  hipExternalMemory_t memory_{};
  void* address_{};
  int fd_ = -1;
};

inline std::expected<vk::net::DeviceMemory, Refusal> import_buffer(ExternalMemoryApi api, OnDevice on_device,
                                                                   std::uintptr_t fd, std::size_t allocation,
                                                                   std::size_t bytes) {
  auto view = std::make_shared<ExternalBuffer>(api, std::move(on_device));
  if (auto ok = view->open(fd, allocation, bytes); !ok) return std::unexpected(ok.error());
  return vk::net::DeviceMemory{view->address(), std::move(view)};
}

}  // namespace kirari::gpu::hip_interop
#endif
