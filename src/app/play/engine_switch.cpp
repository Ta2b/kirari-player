#include "app/play/engine_switch.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <utility>

#include "core/utf8_path.h"
#include "core/zml_plugin.h"
#include "gpu/cuda_mode.h"

namespace kirari::app::play {
namespace {

// Les moteurs des réseaux, dans l'ordre de l'onglet, et le mot de leur bibliothèque.
constexpr std::array<std::pair<std::string_view, core::EngineId>, 7> kLibraries{{
    {"vulkan", core::EngineId::Vulkan},
    {"tensorrt", core::EngineId::TensorRt},
    {"zml", core::EngineId::Zml},
    {"hip", core::EngineId::Hip},
    {"slang", core::EngineId::Slang},
    {"triton", core::EngineId::Triton},
    {"coreml", core::EngineId::Metal},  // le moteur natif du Mac : Core ML derrière MoltenVK
}};
// Les bibliothèques des moteurs retirés (core/engines.h), qu'une ancienne version a pu laisser dans le
// dossier : jamais proposées ni ouvertes.
constexpr std::array<std::string_view, 2> kRetiredLibraries{"migraphx", "openvino"};

constexpr std::size_t usage(core::Usage u) {
  return static_cast<std::size_t>(u);
}

}  // namespace

std::optional<core::EngineId> engine_of_library(const std::filesystem::path& library) {
  const std::string file = core::to_utf8(library.filename());
  if (!file.contains("kirari_backend_")) return std::nullopt;
  for (const auto& [word, id] : kLibraries) {
    if (file.contains(word)) return id;
  }
  return std::nullopt;
}

std::uint32_t cuda_inference_mode(const core::Settings& engines) {
  const auto preferred = engines.preferred[usage(core::Usage::Upscale)];
  if (preferred == core::EngineId::Cuda) return gpu::kCudaNativeInference;
  if (preferred == core::EngineId::TensorRt) return gpu::kTensorRtInference;
  return 0;
}

std::optional<core::EngineId> selected_engine(const std::filesystem::path& library,
                                              const core::Settings& settings) {
  const auto id = engine_of_library(library);
  if (id == core::EngineId::TensorRt && cuda_inference_mode(settings) == gpu::kCudaNativeInference)
    return core::EngineId::Cuda;
  return id;
}

bool backend_available(const std::filesystem::path& path, GpuVendor vendor) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) return false;
  const auto id = engine_of_library(path);
  const auto assets = path.parent_path();
  const auto has_plan = [&](const std::filesystem::path& folder) {
    return std::filesystem::is_regular_file(folder / "plan.tsv", ec) &&
           std::filesystem::is_regular_file(folder / "SHA256SUMS", ec);
  };
  if (id == core::EngineId::Zml) return !zml_plugin_path(path, vendor).empty();
  if (id == core::EngineId::Hip) {
    if (vendor != GpuVendor::Amd && vendor != GpuVendor::Nvidia) return false;
    const std::string_view extension = vendor == GpuVendor::Amd ? ".hsaco" : ".cubin";
    const std::filesystem::directory_iterator end;
    for (std::filesystem::directory_iterator it(assets / "hip", ec); !ec && it != end; it.increment(ec)) {
      if (it->path().extension() == extension && it->is_regular_file(ec)) return true;
    }
    return false;
  }
  if (id == core::EngineId::Slang) return has_plan(assets / "slang");
  if (id == core::EngineId::Triton) {
    if (vendor == GpuVendor::Intel) return has_plan(assets / "triton/arc-a770");
    if (vendor != GpuVendor::Nvidia && vendor != GpuVendor::Amd) return false;
    const std::string_view prefix = vendor == GpuVendor::Nvidia ? "sm" : "gfx";
    const std::filesystem::directory_iterator end;
    for (std::filesystem::directory_iterator it(assets / "triton", ec); !ec && it != end; it.increment(ec)) {
      if (core::to_utf8(it->path().filename()).starts_with(prefix) && has_plan(it->path())) return true;
    }
    return false;
  }
  return true;
}

bool retired_library(const std::filesystem::path& library) {
  const std::string file = core::to_utf8(library.filename());
  return file.contains("kirari_backend_") &&
         std::ranges::any_of(kRetiredLibraries, [&](std::string_view word) { return file.contains(word); });
}

std::vector<core::EngineId> switchable_engines(std::span<const std::filesystem::path> candidates) {
  std::vector<core::EngineId> out;
  for (const auto& entry : kLibraries) {
    const core::EngineId id = entry.second;
    if (!std::ranges::any_of(candidates, [&](const auto& path) { return engine_of_library(path) == id; }))
      continue;
    out.push_back(id);
    if (id == core::EngineId::TensorRt) out.push_back(core::EngineId::Cuda);
  }
  return out;
}

core::Settings using_engine(core::Settings settings, core::EngineId id) {
  if (!std::ranges::contains(settings.chosen_engines, id)) settings.chosen_engines.push_back(id);
  if (!std::ranges::contains(settings.installed_engines, id)) settings.installed_engines.push_back(id);
  settings.preferred[usage(core::Usage::Upscale)] = id;
  settings.preferred[usage(core::Usage::Interpolate)] = id;
  return settings;
}

std::vector<std::filesystem::path> preferred_first(std::vector<std::filesystem::path> candidates,
                                                   const core::Settings& engines) {
  const std::optional<core::EngineId> preferred = engines.preferred[usage(core::Usage::Upscale)];
  if (!preferred) return candidates;
  std::ranges::stable_partition(
      candidates, [&](const auto& path) { return selected_engine(path, engines) == preferred; });
  return candidates;
}

}  // namespace kirari::app::play
