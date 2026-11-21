#pragma once

#include <array>

#include "gpu/zml/pjrt.h"

namespace kirari::gpu::zml {
struct FeatureIo {
  std::shared_ptr<PJRT_Buffer> previous;
  std::shared_ptr<PJRT_Buffer> next;
  bool save = false;
};
class FeatureExecutable {
 public:
  virtual ~FeatureExecutable() = default;
  virtual std::expected<std::shared_ptr<void>, Refusal> run_features(std::span<const vk::net::HostBinding>,
                                                                     std::span<const vk::net::HostBinding>,
                                                                     std::uint64_t, std::uint64_t,
                                                                     FeatureIo*) = 0;
};
std::expected<std::unique_ptr<FeatureExecutable>, Refusal> load_feature_program(
    std::shared_ptr<Client> client, std::span<const std::byte> source, vk::net::IoNames names,
    const std::filesystem::path& path);
std::expected<void, Refusal> prepare_rife_features(const std::shared_ptr<Client>& client,
                                                   const std::filesystem::path& path, Progress& progress);
void load_rife_features(const std::shared_ptr<Client>& client, const std::filesystem::path& path,
                        std::span<const std::byte> source, vk::net::RuntimeNet& dense);
}  // namespace kirari::gpu::zml
