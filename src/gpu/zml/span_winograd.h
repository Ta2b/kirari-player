#pragma once
#include <optional>

#include "gpu/zml/hlo.h"

namespace kirari::gpu::zml {
// F(2×2,3×3) : transformations séparables et un produit matriciel par lot de 16 coefficients.
std::optional<Value> span_winograd(Hlo& h, const ir::Graph& graph, const ir::Node& node,
                                   const std::vector<Value>& values);
}  // namespace kirari::gpu::zml
