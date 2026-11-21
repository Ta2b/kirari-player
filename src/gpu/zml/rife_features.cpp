#include "gpu/zml/rife_features.h"

#include <format>
#include <mutex>

#include "gpu/vulkan/backend/net_build.h"

namespace kirari::gpu::zml {
namespace {
constexpr std::array<std::string_view, 3> kPhases{"froid", "instants", "suite"};

std::filesystem::path sidecar(std::filesystem::path path, std::size_t phase) {
  path += std::format(".rife-{}.mlir", kPhases[phase]);
  return path;
}
vk::net::IoNames bindings(std::size_t phase) {
  vk::net::IoNames io{{"images"}, {"flot", "masque"}};
  if (phase != 0) io.inputs.emplace_back("traits_précédents");
  if (phase != 1) io.outputs.emplace_back("traits");
  return io;
}

class RifeFeatures final : public vk::net::RuntimeRifeFeatures {
 public:
  std::array<std::unique_ptr<FeatureExecutable>, 3> programs;
  void forget() override {
    std::scoped_lock lock(mutex_);
    traits_.reset();
  }
  std::string summary() const override {
    std::scoped_lock lock(mutex_);
    return std::format(
        "cache GPU des traits RIFE : {} paires complètes, {} reprises, {} transitions B vers A", calls_[0],
        calls_[1], calls_[2]);
  }
  std::expected<std::shared_ptr<void>, Refusal> run_device(RifeFeatureRequest request,
                                                           std::span<const vk::net::HostBinding> inputs,
                                                           std::span<const vk::net::HostBinding> outputs,
                                                           std::uint64_t wait, std::uint64_t done) override {
    std::scoped_lock lock(mutex_);
    std::size_t phase = 0;
    if (traits_ && request == RifeFeatureRequest::SamePair) phase = 1;
    if (traits_ && request == RifeFeatureRequest::Following) phase = 2;
    // Un refus laisse le cache vide. Les autres prêts en vol possèdent encore leurs tampons indépendamment.
    FeatureIo io{.previous = std::exchange(traits_, {}), .save = phase != 1};
    if (phase == 0) io.previous.reset();
    auto result = programs[phase]->run_features(inputs, outputs, wait, done, &io);
    if (result) {
      traits_ = io.save ? std::move(io.next) : std::move(io.previous);
      ++calls_[phase];
    }
    return result;
  }

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<PJRT_Buffer> traits_;
  std::array<std::uint64_t, 3> calls_{};
};
constexpr std::string_view kTag = "// kirari-rife-features-v1 ";
std::string marker(const std::array<std::uint64_t, 3>& hashes) {
  return std::format("{}{:016x} {:016x} {:016x}\n", kTag, hashes[0], hashes[1], hashes[2]);
}
}  // namespace

std::expected<void, Refusal> prepare_rife_features(const std::shared_ptr<Client>& client,
                                                   const std::filesystem::path& path, Progress& progress) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(sidecar(path, 0), ec)) return {};
  std::array<std::uint64_t, 3> hashes{};
  for (std::size_t phase = 0; phase < kPhases.size(); ++phase) {
    if (progress.stop.stop_requested())
      return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation du cache de traits RIFE annulée"});
    const auto file = sidecar(path, phase);
    auto bytes = vk::net::read_file(file);
    if (!bytes) return std::unexpected(bytes.error());
    hashes[phase] = vk::net::fnv1a(*bytes);
    auto result = compile(client, *bytes, bindings(phase), file, &progress);
    if (!result) return std::unexpected(result.error());
  }
  auto dense = vk::net::read_file(path);
  if (!dense) return std::unexpected(dense.error());
  const auto tag = marker(hashes);
  const auto prefix = std::as_bytes(std::span(tag));
  dense->insert(dense->begin(), prefix.begin(), prefix.end());
  return vk::net::write_file(path, *dense);
}
void load_rife_features(const std::shared_ptr<Client>& client, const std::filesystem::path& path,
                        std::span<const std::byte> source, vk::net::RuntimeNet& dense) {
  const std::string_view text(reinterpret_cast<const char*>(source.data()), source.size());
  if (!text.starts_with(kTag)) return;
  std::array<std::uint64_t, 3> hashes{};
  auto cache = std::make_unique<RifeFeatures>();
  for (std::size_t phase = 0; phase < kPhases.size(); ++phase) {
    const auto file = sidecar(path, phase);
    auto bytes = vk::net::read_file(file);
    if (!bytes) return;
    hashes[phase] = vk::net::fnv1a(*bytes);
    auto exe = load_feature_program(client, *bytes, bindings(phase), file);
    if (!exe) return;
    cache->programs[phase] = std::move(*exe);
  }
  if (!text.starts_with(marker(hashes))) return;
  dense.rife_features = std::move(cache);
}
}  // namespace kirari::gpu::zml
