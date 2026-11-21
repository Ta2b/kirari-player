#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "gpu/ir/ir.h"

namespace kirari::gpu::zml {
struct SpanStrips {
  std::int64_t height, width, core, halo, window;
};
std::optional<SpanStrips> span_strips_plan(const ir::Dims& dims, std::size_t blocks);
// Le module de la fenêtre conserve tous les poids et ses fonctions privées.
std::optional<std::string> wrap_span_strips(std::string tile_module, const SpanStrips& plan);
}  // namespace kirari::gpu::zml
