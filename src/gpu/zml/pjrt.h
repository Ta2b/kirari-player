#pragma once
#include <optional>
#include <xla/pjrt/c/pjrt_c_api.h>

#include "core/dynamic_library.h"
#include "gpu/zml/runtime.h"

namespace kirari::gpu::zml {
std::expected<void, Refusal> checked(const PJRT_Api& api, PJRT_Error* error, std::string_view call);
void discard(const PJRT_Api& api, PJRT_Error* error);

// Le client est partagé entre tous les réseaux du lecteur. Il survit aux exécutables et aux étapes en vol.
class CudaIo;
struct Client {
  std::shared_ptr<CudaIo> cuda;
  DynamicLibrary library;
  const PJRT_Api& api;
  PJRT_Client* handle = nullptr;
  PJRT_Device* device = nullptr;
  int device_id = 0;
  std::optional<int> local_id;
  std::string name, version;
  Client(DynamicLibrary lib, const PJRT_Api& table) : library(std::move(lib)), api(table) {}
  ~Client();
};
bool can_cache_executable(const PJRT_Api& api);
std::filesystem::path executable_cache(const std::filesystem::path& path, std::span<const std::byte> source,
                                       int device);
std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> compile(
    std::shared_ptr<Client> client, std::span<const std::byte> program, vk::net::IoNames names,
    const std::filesystem::path& path, Progress* preparation = nullptr, bool cached_only = false);
}  // namespace kirari::gpu::zml
