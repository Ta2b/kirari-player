#pragma once

#include "gpu/zml/motion_groups.h"

namespace kirari::gpu::zml {
// Appariement bidirectionnel d'une paire GMFlow aux dimensions de lecture qualifiées.
std::optional<MotionGroup> gmflow_global_match(const ir::Graph& graph);
}  // namespace kirari::gpu::zml
