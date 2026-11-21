#pragma once

#include <optional>

#include "gpu/ir/ir.h"

namespace kirari::gpu::zml {
// Les deux Reshape remettent les deux images du lot sur l'axe des canaux avant la réduction RIFE.
struct CompactWarpPlan {
  const ir::Node& warp;
  const ir::Node& split;
  const ir::Node& join;
  std::int64_t factor;
  bool cast_warp;
  bool cast_planes;
};

std::optional<CompactWarpPlan> compact_warp_plan(const ir::Graph& graph, const ir::Node& resize,
                                                 float scale_h, float scale_w, bool include_half = false);
#ifdef KIRARI_ZML_LINEAR_WARP
// Le tenseur de pixels FP16 du warp RGB + traits qui précède le bloc RIFE pleine taille.
std::optional<ir::TensorId> linear_warp_source(const ir::Graph& graph, const ir::Node& warp);
#endif
}  // namespace kirari::gpu::zml
