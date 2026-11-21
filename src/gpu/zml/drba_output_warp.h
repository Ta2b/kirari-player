#pragma once

#include <optional>

#include "gpu/zml/motion_groups.h"

namespace kirari::gpu::zml {
std::optional<MotionGroup> drba_output_warp(const ir::Graph& graph);
// Projection FP16 de l’image extérieure du trio, partagée par les deux réductions et le warp final.
ir::TensorId drba_shared_projection(const ir::Graph& graph, const MotionGroup& finish);
}  // namespace kirari::gpu::zml
