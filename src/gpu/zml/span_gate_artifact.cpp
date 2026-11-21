#include "gpu/zml/span_gate_artifact.h"

#include <format>

#include "core/utf8_path.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/zml/sdk_process.h"

namespace kirari::gpu::zml {
namespace {
constexpr std::string_view kPrefix = "// kirari-span-portable-v1 ";
std::filesystem::path portable_path(std::filesystem::path path) {
  return path += ".portable";
}
std::string header(std::span<const std::byte> dense) {
  return std::format("{}{:016x}\n", kPrefix, vk::net::fnv1a(dense));
}
std::string_view as_text(std::span<const std::byte> code) {
  return {reinterpret_cast<const char*>(code.data()), code.size()};
}
}  // namespace

std::expected<bool, Refusal> emit_span_gates(std::vector<std::string> arguments,
                                             const std::filesystem::path& staged,
                                             const SpanGateProfile& profile, bool fused_projection,
                                             Progress& progress) {
  auto portable = arguments;
  const auto sidecar = portable_path(staged);
  portable.at(3) = core::to_utf8(sidecar);
  auto built = run_sdk_frontend(std::move(portable), progress);
  if (!built || !*built) return built;
  auto dense = vk::net::read_file(sidecar);
  if (!dense) return std::unexpected(dense.error());
  if (fused_projection) {
    std::erase(arguments, "--projection-strips");
    arguments.emplace_back("--span-balanced-triton-projection");
  }
  arguments.emplace_back(profile.option);
  auto candidate = run_sdk_frontend(std::move(arguments), progress);
  if (!candidate && candidate.error().code != RefusalCode::BuilderFailed) return candidate;
  if (progress.stop.stop_requested())
    return refuse(RefusalCode::Cancelled, "préparation des portes SPAN annulée");
  if (candidate && *candidate) {
    auto region = vk::net::read_file(staged);
    if (!region) return std::unexpected(region.error());
    const auto expected = fused_projection ? kSpanProjectionModule : profile.module;
    if (span_gate_module(as_text(*region)) == expected) {
      auto text = header(*dense);
      text.append(as_text(*region));
      if (auto saved = vk::net::write_file(staged, std::as_bytes(std::span(text))); !saved)
        return std::unexpected(saved.error());
      return true;
    }
  }
  if (progress.report) progress.report(0.5F, "le frontal utilise le graphe SPAN portable");
  if (auto saved = vk::net::write_file(staged, *dense); !saved) return std::unexpected(saved.error());
  return true;
}

std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> compile_span_or_portable(
    const std::shared_ptr<Client>& client, std::span<const std::byte> source, const vk::net::IoNames& names,
    const std::filesystem::path& path, Progress* preparation) {
  auto result = compile(client, source, names, path, preparation);
  if (result || !as_text(source).starts_with(kPrefix)) return result;
  const auto reason = result.error().code;
  if (reason != RefusalCode::SdkRejected && reason != RefusalCode::VendorCallFailed) return result;
  if (preparation && preparation->stop.stop_requested())
    return refuse(RefusalCode::Cancelled, "compilation des portes SPAN annulée");
  auto dense = vk::net::read_file(portable_path(path));
  if (!dense || !as_text(source).starts_with(header(*dense))) return result;
  if (preparation && preparation->report)
    preparation->report(
        0.7F, std::format("compilation du SPAN portable après refus PJRT : {}", result.error().detail));
  auto fallback = compile(client, *dense, names, path, preparation);
  // Sous le verrou de préparation, associer la source publiée au binaire réellement compilé.
  if (fallback && preparation) {
    if (auto saved = vk::net::write_file(path, *dense); !saved) return std::unexpected(saved.error());
  }
  return fallback;
}
}  // namespace kirari::gpu::zml
