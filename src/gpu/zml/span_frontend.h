#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "gpu/vulkan/backend/net_runtime.h"

namespace kirari::gpu::zml {
struct SpanSdk {
  bool available = false;
  bool balanced_strips = false;
  bool performance_strips = false;

  bool pixel_strips = false;
  bool balanced_gates = false;
  bool fused_projection = false;
  bool performance_gates = false;

  std::string_view projection_option(int channels, std::size_t blocks, std::int64_t height,
                                     std::int64_t width) const {
    if (!available || height != 1080 || width != 1920) return {};
    if (performance_strips && channels == 48 && blocks == 5)
      return pixel_strips ? "--projection-pixel-strips" : "--projection-strips";
    if (balanced_strips && channels == 64 && blocks == 8) return "--projection-strips";
    return {};
  }
};

// false laisse le lecteur importer normalement un autre modèle ou une autre forme.
std::expected<bool, Refusal> build_recognized_span(const ModelSpec& model,
                                                   std::span<const vk::net::InputShape> inputs,
                                                   const std::filesystem::path& staged, SpanSdk sdk,
                                                   Progress& progress);
}  // namespace kirari::gpu::zml
