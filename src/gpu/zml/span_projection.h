#pragma once

#include <optional>

#include "gpu/zml/hlo.h"

namespace kirari::gpu::zml {
// Évite de matérialiser les quatre cartes concaténées avant la projection 1×1.
std::optional<Value> split_projection(Hlo& hlo, const ir::Graph& graph, const ir::Node& node,
                                      const std::vector<Value>& values);
}  // namespace kirari::gpu::zml
