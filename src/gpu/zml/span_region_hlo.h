#pragma once
#include <optional>
#include <string>

namespace kirari::gpu::zml {
std::optional<std::string> wrap_span_regions(std::string tile, int h, int w);
}  // namespace kirari::gpu::zml
