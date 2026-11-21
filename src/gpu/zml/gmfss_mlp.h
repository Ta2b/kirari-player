#pragma once

#include "gpu/zml/motion_groups.h"

namespace kirari::gpu::zml {
// Premier MLP au quart du GMFlow s05 à 1920 × 1152, une seule sortie et quatre paramètres appris.
std::optional<MotionGroup> gmflow_quarter_mlp(const ir::Graph& graph);
}  // namespace kirari::gpu::zml
